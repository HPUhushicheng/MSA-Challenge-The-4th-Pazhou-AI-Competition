import os
import time
import csv
import subprocess
import sys
from tqdm import tqdm
import shutil  
from faster_whisper import WhisperModel

def log_message(log_file, message, print_to_console=False):
    """将消息写入日志文件，可选是否同时打印到控制台"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    if print_to_console:
        print(message)

def initialize_csv(csv_file):
    """初始化CSV文件并写入表头"""
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['文件名称', '语言', '字幕', '处理时间(秒)'])

def write_to_csv(csv_file, filename, language, subtitle, process_time):
    """将处理结果写入CSV文件"""
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([filename, language, subtitle, f"{process_time:.2f}"])
        
def process_audio(model, audio_path, log_file, csv_file):
    """使用已加载的模型处理单个音频文件，并按语言分类复制文件"""
    if not os.path.exists(audio_path):
        log_message(log_file, f"警告：音频文件 {audio_path} 不存在，已跳过")
        return False
    
    # 记录单个文件的转录时间
    start_time = time.time()
    segments, info = model.transcribe(audio_path, beam_size=5)
    end_time = time.time()
    
    filename = os.path.basename(audio_path)
    log_message(log_file, f"\n===== 处理文件: {filename} =====")
    log_message(log_file, f"检测到的原始语言: {info.language}，概率: {info.language_probability}")
    
    # 语言归一化处理
    normalized_language = info.language if info.language == 'zh' else 'en'
    log_message(log_file, f"归一化后的语言: {normalized_language}")
    
    full_subtitle = []
    for segment in segments:
        segment_text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        log_message(log_file, segment_text)
        full_subtitle.append(segment_text)
    
    # 合并所有字幕片段为一个字符串
    subtitle_str = '\n'.join(full_subtitle)
    
    processing_time = end_time - start_time
    log_message(log_file, f"单个文件处理耗时: {processing_time:.2f}秒")
    
    # 写入CSV文件（使用归一化后的语言）
    write_to_csv(csv_file, filename, normalized_language, subtitle_str, processing_time)
    
    # === 新增：复制文件到对应语言文件夹 ===
    try:
        audio_dir = os.path.dirname(audio_path)
        target_folder = os.path.join(audio_dir, normalized_language)
        
        # 创建目标文件夹
        os.makedirs(target_folder, exist_ok=True)
        
        target_path = os.path.join(target_folder, filename)
        
        if not os.path.exists(target_path):
            shutil.copy(audio_path, target_path)
            log_message(log_file, f"已复制文件到: {target_folder}/")
        else:
            log_message(log_file, f"文件已存在，跳过复制: {target_path}")
            
    except Exception as e:
        log_message(log_file, f"复制文件时出错 {audio_path}: {e}")
    
    return True

def batch_process_audio(audio_dir):
    try:
        # 创建固定名称的日志文件和CSV文件
        log_file = "transcription_log.txt"
        csv_file = "transcription_results.csv"
        
        # 初始化日志和CSV文件（清空之前的内容）
        with open(log_file, 'w', encoding='utf-8') as f:
            pass  # 清空日志文件
        initialize_csv(csv_file)
        
        # 1. 模型初始化（放在循环外，只执行一次）
        model_load_start = time.time()
        log_message(log_file, "开始加载音频字幕模型...", True)
        
        model = WhisperModel(
            "models/faster-whisper-large-v3-turbo",
            device="cuda", 
            compute_type="float32"
        )
        
        model_load_end = time.time()
        load_time = model_load_end - model_load_start
        log_message(log_file, f"模型加载完成，耗时: {load_time:.2f}秒", True)
        
        # 2. 获取需要处理的音频文件列表（这里假设处理目录下的所有mp4文件）
        audio_files = [
            os.path.join(audio_dir, f) 
            for f in os.listdir(audio_dir) 
            if f.lower().endswith(".mp4")
        ]
        
        if not audio_files:
            log_message(log_file, f"在目录 {audio_dir} 中未找到mp4文件", True)
            return
        
        log_message(log_file, f"发现 {len(audio_files)} 个mp4文件需要处理", True)
        
        # 3. 循环处理多个音频文件（复用已加载的模型），使用进度条
        total_start = time.time()
        
        # 创建进度条
        with tqdm(total=len(audio_files), desc="处理进度", unit="文件") as pbar:
            success_count = 0
            for audio_file in audio_files:
                # 更新进度条描述，显示当前处理的文件名
                filename = os.path.basename(audio_file)
                pbar.set_postfix_str(f"正在处理: {filename[:30]}...")
                
                # 处理音频文件
                success = process_audio(model, audio_file, log_file, csv_file)
                if success:
                    success_count += 1
                
                # 更新进度条
                pbar.update(1)
        
        total_end = time.time()
        total_time = total_end - total_start
        processing_time = total_end - model_load_end
        
        summary = f"\n===== 音频字幕批量处理完成 ====="
        summary += f"\n处理文件总数: {len(audio_files)}个"
        summary += f"\n成功处理: {success_count}个"
        summary += f"\n总耗时（含模型加载）: {total_time:.2f}秒"
        #summary += f"\n实际转录总耗时（不含模型加载）: {processing_time:.2f}秒"
        summary += f"\n日志已保存至: {os.path.abspath(log_file)}"
        summary += f"\n结果已保存至，wps打开不会乱码: {os.path.abspath(csv_file)}"
        
        log_message(log_file, summary, True)
        return audio_dir  # 返回音频目录用于后续处理

    except Exception as e:
        error_msg = f"处理出错: {e}"
        print(error_msg)
        if 'log_file' in locals():
            log_message(log_file, error_msg)
        return None

def wait_for_folders(audio_dir, max_wait_seconds=60, check_interval=5):
    """等待zh和en文件夹存在，超时则返回False"""
    zh_folder = os.path.join(audio_dir, "zh")
    en_folder = os.path.join(audio_dir, "en")
    
    waited = 0
    while waited < max_wait_seconds:
        # 检查文件夹是否存在
        zh_exists = os.path.exists(zh_folder) and os.path.isdir(zh_folder)
        en_exists = os.path.exists(en_folder) and os.path.isdir(en_folder)
        
        # 至少有一个文件夹存在就可以继续
        if zh_exists or en_exists:
            print(f"检测到语言文件夹存在 (zh: {zh_exists}, en: {en_exists})，继续处理...")
            return True
            
        # 等待一段时间后再次检查
        wait_time = min(check_interval, max_wait_seconds - waited)
        print(f"等待语言文件夹创建... ({waited}/{max_wait_seconds}秒)")
        time.sleep(wait_time)
        waited += wait_time
    
    # 超时
    print(f"超过最大等待时间({max_wait_seconds}秒)，未检测到语言文件夹")
    return False

if __name__ == "__main__":
    # 音频文件所在的目录
    audio_directory = "test_data"
    audio_dir = batch_process_audio(audio_directory)
    
    if audio_dir:
        # 等待文件夹存在
        if wait_for_folders(audio_dir):
            # 自动调用Multi-infer.py处理中文音频
            print("\n=====开始调用multi-infer.py处理中文音频====...")
            zh_command = [
                sys.executable,  # 使用当前Python解释器
                "multi-infer-test.py",
                "--audio_path", os.path.join(audio_dir, "zh"),  # 中文音频目录
                "--output_csv", "zh-results.csv"
            ]
            subprocess.run(zh_command, check=True)
            
            # 调用Multi-infer.py处理英文音频
            print("\n====开始调用multi-infer.py处理英文音频====...")
            en_command = [
                sys.executable,
                "multi-infer-test.py",
                "--audio_path", os.path.join(audio_dir, "en"),  # 英文音频目录
                "--output_csv", "en-results.csv",
                "--configs","configs/bi_lstm_new_cmu0813.yml",
                "--model_path", "models/new-cmu-0813-model/BiLSTM_Emotion2Vec/best_model"
            ]
            subprocess.run(en_command, check=True)
            
            print("\n所有处理已完成！")
        else:
            print("未找到语言文件夹，无法继续处理")
    else:
        print("音频处理过程中出现错误，无法继续")

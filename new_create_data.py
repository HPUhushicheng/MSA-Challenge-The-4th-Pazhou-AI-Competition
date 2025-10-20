import os
import time
import numpy as np
from mser.trainer import MSERTrainer

# 全局配置
LABELS = ["NEUT", "SNEG", "SPOS", "WNEG", "WPOS"]  # 必须与文件夹名称严格对应
AUDIO_EXT = '.wav'  # 支持的音频格式

def validate_folder_structure(root_dir):
    """验证目录结构是否符合标签要求"""
    if not os.path.exists(root_dir):
        raise ValueError(f"根目录不存在: {root_dir}")
    
    missing = [label for label in LABELS 
               if not os.path.exists(os.path.join(root_dir, label))]
    if missing:
        raise ValueError(f"缺失必须的标签文件夹: {missing}\n"
                        f"目录结构应包含: {LABELS}")

def generate_label_file(output_path):
    """生成标签描述文件"""
    with open(os.path.join(output_path, 'label_list.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(LABELS))

def get_data_list(audio_path, list_path):
    """生成原始数据列表（自动划分train/test）"""
    validate_folder_structure(audio_path)
    generate_label_file(list_path)
    
    train_list = []
    test_list = []
    
    for label_id, label in enumerate(LABELS):
        class_dir = os.path.join(audio_path, label)
        files = [f for f in os.listdir(class_dir) if f.endswith(AUDIO_EXT)]
        
        for idx, file in enumerate(sorted(files)):  # 排序保证可复现
            # 先处理路径再放入f-string
            file_path = os.path.join(class_dir, file).replace('\\', '/')
            if idx % 10 == 0:  # 10%作为测试集
                test_list.append(f"{file_path}\t{label_id}\n")
            else:
                train_list.append(f"{file_path}\t{label_id}\n")
    
    # 写入文件
    with open(os.path.join(list_path, 'train_list.txt'), 'w') as f:
        f.writelines(train_list)
    with open(os.path.join(list_path, 'test_list.txt'), 'w') as f:
        f.writelines(test_list)

def process_new_data(new_dir, output_path):
    """处理新增数据（不自动划分）"""
    validate_folder_structure(new_dir)
    new_data = []
    
    for label_id, label in enumerate(LABELS):
        class_dir = os.path.join(new_dir, label)
        if not os.path.exists(class_dir):
            continue
            
        files = [f for f in os.listdir(class_dir) if f.endswith(AUDIO_EXT)]
        for f in sorted(files):
            # 先处理路径再放入f-string，修复语法错误
            file_path = os.path.join(class_dir, f).replace('\\', '/')
            new_data.append(f"{file_path}\t{label_id}\n")
    
    # 保存新数据临时列表
    with open(os.path.join(output_path, 'new_data_list.txt'), 'w') as f:
        f.writelines(new_data)
    return len(new_data)

def interactive_merge():
    """交互式合并数据"""
    # 读取现有数据量
    def count_lines(file):
        return sum(1 for _ in open(file)) if os.path.exists(file) else 0
    
    orig_train = count_lines('dataset/train_list.txt')
    orig_test = count_lines('dataset/test_list.txt')
    new_data = count_lines('dataset/new_data_list.txt')
    
    print(f"\n当前数据统计:")
    print(f"原始训练集: {orig_train} 条")
    print(f"原始测试集: {orig_test} 条")
    print(f"新增数据: {new_data} 条")
    
    # 用户控制分配
    while True:
        try:
            ratio = float(input("👉 请输入新增数据分配到测试集的比例 (0-1): "))
            if 0 <= ratio <= 1:
                break
            print("请输入0-1之间的数字！")
        except ValueError:
            print("请输入有效数字！")
    
    # 执行分配
    with open('dataset/new_data_list.txt') as f:
        all_new = f.readlines()
    
    split_idx = int(len(all_new) * ratio)
    new_test = all_new[:split_idx]
    new_train = all_new[split_idx:]
    
    # 追加到原文件
    with open('dataset/test_list.txt', 'a') as f:
        f.writelines(new_test)
    with open('dataset/train_list.txt', 'a') as f:
        f.writelines(new_train)
    
    print(f"分配结果: {len(new_test)}条到测试集, {len(new_train)}条到训练集")

def create_standard(config_file):
    """生成归一化文件（带确认）"""
    # 定义count_lines函数用于统计行数
    def count_lines(file):
        return sum(1 for _ in open(file)) if os.path.exists(file) else 0
    
    print("\n即将执行数据归一化...")
    print("请确认以下文件已准备就绪:")
    print(f"- train_list.txt (共 {count_lines('dataset/train_list.txt')} 条)")
    print(f"- test_list.txt (共 {count_lines('dataset/test_list.txt')} 条)")
    
    if input("✅ 是否继续？(y/n) ").lower() == 'y':
        MSERTrainer(configs=config_file).get_standard_file()
        print("归一化完成！")
    else:
        print("❌ 已取消归一化")

def main():
    try:
        print("==== 数据预处理开始 ====")
        
        # 第一阶段：处理原始数据
        print("\n[阶段1] 处理原始数据...")
        get_data_list('dataset/audio', 'dataset')
        
        # 第二阶段：交互式添加新数据
        print("\n[阶段2] 添加新数据")
        input("👉 请将新数据按标签放入 dataset/new_data/ 下的对应子目录后按回车继续...")
        
        if not os.path.exists('dataset/new_data'):
            raise FileNotFoundError("未找到 new_data 目录")
            
        new_count = process_new_data('dataset/new_data', 'dataset')
        print(f"检测到 {new_count} 条新增数据")
        
        # 第三阶段：合并数据
        if new_count > 0:
            print("\n[阶段3] 数据合并")
            interactive_merge()
        
        # 第四阶段：归一化
        print("\n[阶段4] 数据归一化")
        create_standard('configs/bi_lstm_0815.yml')
        
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
    finally:
        print("\n==== 处理结束 ====")

if __name__ == '__main__':
    main()

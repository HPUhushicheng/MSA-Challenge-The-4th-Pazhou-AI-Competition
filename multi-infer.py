import argparse
import functools
import os
import csv
from pathlib import Path

from mser.predict import MSERPredictor
from mser.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/bi_lstm_0815.yml',   '注意注意。。。。。。。。。。。修改配置文件')
add_arg('use_ms_model',     str,    None,                    '使用ModelScope上公开Emotion2vec的模型')
add_arg('use_gpu',          bool,   True,                    '是否使用GPU预测')
add_arg('audio_path',       str,    'test_data/zh',      '音频路径或文件夹路径')
add_arg('num_files',        int,    -1,                      '要处理的音频数量（-1表示全部）')
add_arg('start_index',      int,    0,                       '起始文件索引(从0开始)')
add_arg('model_path',       str,    'models/0815-ch/BiLSTM_Emotion2Vec/best_model', '训练的单独ch模型')
add_arg('output_csv',       str,    '0816zh-results.csv',       '输出结果CSV文件路径')
args = parser.parse_args()
print_arguments(args=args)

# 获取识别器
predictor = MSERPredictor(configs=args.configs,
                          use_ms_model=args.use_ms_model,
                          model_path=args.model_path,
                          use_gpu=args.use_gpu)

def process_audio_file(audio_path):
    label, score = predictor.predict(audio_data=audio_path)
    #print(f'音频：{audio_path} 的预测结果标签为：{label}，得分：{score}')
    print(f'音频：{audio_path} 的预测结果标签为：{label}')
    return label, score

if os.path.isdir(args.audio_path):
    # 处理文件夹中的音频文件
    audio_files = sorted([str(f) for f in Path(args.audio_path).glob('*') 
                         if f.suffix.lower() in ['.wav','.mp4', '.mp3', '.flac', '.ogg']])
    
    total_files = len(audio_files)
    
    # 确定要处理的文件范围
    if args.num_files > 0:
        # 使用num_files参数
        end_index = args.start_index + args.num_files - 1
        if end_index >= total_files:
            end_index = total_files - 1
    else:
        # 处理从start_index开始的所有文件
        end_index = total_files - 1
    
    if args.start_index < 0:
        args.start_index = 0
    
    if args.start_index > end_index:
        raise ValueError("起始索引不能大于结束索引")
    
    audio_files = audio_files[args.start_index : end_index + 1]
    
    print(f"开始批量处理 {len(audio_files)} 个音频文件 (从第{args.start_index}到第{end_index}个)...")
    
    results = []
    for audio_file in audio_files:
        try:
            label, score = process_audio_file(audio_file)
            file_id = os.path.basename(audio_file)
            results.append({
                'ID': file_id,
                'Label': label,
                # 'score': float(score)  # 如果需要分数可以取消注释
            })
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
    
    # 保存为CSV文件
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'Label']  # 如果需要分数可以添加 'score'
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n结果已保存到 {args.output_csv}")

else:
    # 处理单个音频文件
    label, score = process_audio_file(args.audio_path)
    file_id = os.path.basename(args.audio_path)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['ID', 'Label']  # 如果需要分数可以添加 'score'
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({
            'ID': file_id,
            'Label': label,
            # 'score': float(score)  # 如果需要分数可以取消注释
        })
    print(f"\n结果已保存到 {args.output_csv}")

# import argparse
# import functools
# import os
# import csv
# from pathlib import Path
# from datetime import datetime

# from mser.predict import MSERPredictor
# from mser.utils.utils import add_arguments, print_arguments

# parser = argparse.ArgumentParser(description=__doc__)
# add_arg = functools.partial(add_arguments, argparser=parser)
# add_arg('configs',          str,    'configs/bi_lstm.yml',   '配置文件')
# add_arg('use_ms_model',     str,    None,                    '使用ModelScope上公开Emotion2vec的模型')
# add_arg('use_gpu',          bool,   True,                    '是否使用GPU预测')
# add_arg('audio_path',       str,    'dataset/test.wav',      '音频路径或文件夹路径')
# add_arg('num_files',        int,    -1,                      '要处理的音频数量（-1表示全部）')
# add_arg('model_path',       str,    'models/cmu-model/BiLSTM_Emotion2Vec/best_model/', '训练的单独ch模型')
# add_arg('output_csv',       str,    'cmu-results.csv',           '输出结果CSV文件路径')
# args = parser.parse_args()
# print_arguments(args=args)

# # 获取识别器
# predictor = MSERPredictor(configs=args.configs,
#                           use_ms_model=args.use_ms_model,
#                           model_path=args.model_path,
#                           use_gpu=args.use_gpu)

# def process_audio_file(audio_path):
#     label, score = predictor.predict(audio_data=audio_path)
#     print(f'音频：{audio_path} 的预测结果标签为：{label}，得分：{score}')
#     return label, score

# if os.path.isdir(args.audio_path):
#     # 处理文件夹中的音频文件
#     audio_files = sorted([str(f) for f in Path(args.audio_path).glob('*') 
#                          if f.suffix.lower() in ['.wav','.mp4', '.mp3', '.flac', '.ogg']])
    
#     if args.num_files > 0:
#         audio_files = audio_files[:args.num_files]
    
#     print(f"开始批量处理 {len(audio_files)} 个音频文件...")
    
#     results = []
#     for audio_file in audio_files:
#         try:
#             label, score = process_audio_file(audio_file)
#             # 提取文件名作为ID
#             file_id = os.path.basename(audio_file)
#             results.append({
#                 'ID': file_id,
#                 'label': label,
#               #  'score': float(score)
#             })
#         except Exception as e:
#             print(f"处理文件 {audio_file} 时出错: {str(e)}")
    
#     # 保存为CSV文件
#     with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         #fieldnames = ['ID', 'label', 'score']
#         fieldnames = ['ID', 'label']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         writer.writeheader()
#         for result in results:
#             writer.writerow(result)
    
#     print(f"\n结果已保存到 {args.output_csv}")

# else:
#     # 处理单个音频文件
#     label, score = process_audio_file(args.audio_path)
#     # 保存单个结果
#     file_id = os.path.basename(args.audio_path)
#     with open(args.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['ID', 'label', 'score']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         writer.writeheader()
#         writer.writerow({
#             'ID': file_id,
#             'label': label,
#            # 'score': float(score)
#         })
#     print(f"\n结果已保存到 {args.output_csv}")
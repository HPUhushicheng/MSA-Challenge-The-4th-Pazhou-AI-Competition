# 本地机器配置
系统配置
PyTorch  2.5.1
Python  3.12
CUDA  12.4
(ubuntu22.04)

# conda msa虚拟环境 配置
PyTorch: 2.8.0
CUDA: 12.8
cuDNN: 91002
python=3.10 
(ubuntu22.04)

# 创建conda msa环境以及安装依赖
conda create -n msa python=3.10 -y
activate msa
cd MSA_tijiao_code
pip install -r requirements.txt
sudo apt update 
sudo apt install ffmpeg -y  # ubuntu




# 准备工作
测试环境中 MSA_tijiao_code为当前目录
请将测试集视频复制一份放入MSA_tijiao_code/Test_Data
重要：以下提到的BERT模型全部填写绝对路径，按照你当前的系统文件的绝对路径写，BERT模型放置在R1-Omni/models中，以下为测试环境路径
需要修改以下代码中的模型路径
```bash
./R1-Omni/humanomni/model/humanomni_arch.py:83:        bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased" #修改为你当前文件绝对路径
./R1-Omni/src/r1-v/src/open_r1/trainer/humanOmni_grpo_trainer.py:297:        bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased"  #修改为你当前文件绝对路径
./R1-Omni/models/R1-Omni-0.5B/config.json  23和31行也填写为绝对路径
"mm_audio_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/whisper-large-v3-turbo" #修改为你当前文件绝对路径
"mm_vision_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/siglip-base-patch16-224" #修改为你当前文件绝对路径
```

# 运行
```py
cd MSA_code
python test_script.py
```

# 结果
>R1-Omni的视频情感分析think COT思维链包存在Test_Supplements/video_emotion_results，字幕结果和日志会保存在Test_Supplements/transcription_results.csv与Test_Supplements/transcription_log.txt中
>最终的监测结果放在Test_Results/combined_results.csv中



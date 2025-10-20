# 摘要
本⽂详细介绍了⼀种⾯向“跨语⾔多模态可解释情感识别”竞赛所设计的端到端深度学习模
型。该模型融合了R1-Omni、emotion2vec、Whisper-large-v3-turbo、SigLIP-base-patch16-224以及BERT-base-uncased等多种先进预训练模型，构建了⼀个兼具跨语⾔处理
能⼒、多模态信息融合能⼒与决策可解释性的统⼀情感识别框架。通过引⼊强化学习可验证
奖励（Reinforcement Learning with Verifiable Rewards, RLVR）机制，模型实现了对⽂本、
语⾳与视觉模态特征的动态融合与归因分析，在中英⽂视频情感识别任务上展现出优异的准
确率与跨语⾔泛化性能。


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
```sh
conda create -n msa python=3.10 -y
activate msa
cd MSA_code
pip install -r requirements.txt
sudo apt update 
sudo apt install ffmpeg -y  # ubuntu
```



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

#  参考⽂献
1. R1-Omni: Reinforcement Learning with Verifiable Rewards for Omni-modal Emotion
Recognition
2. emotion2vec: Self-supervised Pre-training for Universal Speech Emotion
Representation
3. Radford, A., et al. (2019). Improving language understanding by generative pre-training.
4. Wang, X., et al. (2023). SigLIP: Signature-based Language-Image Pre-training.
5. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Supervised
Training.

# 结果
>R1-Omni的视频情感分析think COT思维链包存在Test_Supplements/video_emotion_results，字幕结果和日志会保存在Test_Supplements/transcription_results.csv与Test_Supplements/transcription_log.txt中
>最终的监测结果放在Test_Results/combined_results.csv中



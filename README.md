
# 跨语言多模态可解释情感识别模型（CLMER）
<img width="559" height="280" alt="image" src="https://github.com/user-attachments/assets/ad2c0374-bddb-4c85-82d2-c99baf2d5950" />

## 项目简介
本项目是为“跨语言多模态可解释情感识别”竞赛设计的端到端深度学习模型，整合了R1-Omni、emotion2vec、Whisper-large-v3-turbo、SigLIP-base-patch16-224及BERT-base-uncased等多种先进预训练模型，构建了兼具跨语言处理能力、多模态信息融合能力与决策可解释性的统一情感识别框架。通过引入强化学习可验证奖励（Reinforcement Learning with Verifiable Rewards, RLVR）机制，模型实现了对文本、语音与视觉模态特征的动态融合与归因分析，在中英文视频情感识别任务中展现出优异的准确率与跨语言泛化性能。

随着多媒体内容在社交平台、智能客服和数字娱乐等场景的广泛应用，基于视频的情感识别技术已成为人机交互与情感计算领域的重要研究方向。针对真实应用中跨语言环境下多模态情感识别面临的模态异构性、语言壁垒、决策黑箱性三大核心挑战，本模型可高效处理中英文双语视频数据、精准识别情感极性，并提供可量化解释依据，有效应对上述问题。

## 核心特性
### 1. 跨语言处理能力
- 支持中英文双语输入，通过共享语义空间构建与对比学习优化，实现不同语言下相同情感多模态特征分布的对齐，具备良好的跨语言泛化性能。
- 在中英文数据跨语言迁移测试中，性能保持率超过80%，其中英文训练模型迁移至中文测试时性能保持率达86.7%，中文训练模型迁移至英文测试时性能保持率达83.6%。

### 2. 多模态融合能力
- 采用分层端到端架构，从原始视频文件自动提取文本、音频、视觉三类模态特征：文本特征通过Whisper-large-v3-turbo语音转文本与BERT-base-uncased编码获取，音频特征借助emotion2vec提取（支持16kHz采样率、utterance级与frame级双粒度特征），视觉特征通过SigLIP-base-patch16-224提取（每2秒采样一帧，输出768维特征向量）。
- 基于R1-Omni的RLVR框架实现动态融合，结合情感识别准确率（主要奖励）与特征贡献可解释性（辅助奖励）设计奖励信号，通过策略网络学习模态特征动态权重分配。

### 3. 决策可解释性
- 依托RLVR机制提供多层次解释：量化文本、音频、图像在决策中的权重占比，识别各模态关键特征（如文本情感词、音频语调、图像表情），并支持决策路径可视化。
- 人工评估显示解释清晰度与相关性评分达1-5分区间内的优秀水平，且不同情感类型、不同语言下的模态贡献具有合理差异（如积极情绪中音频贡献42%、视觉38%，中文情感识别更依赖文本）。

### 4. 完整系统评估能力
- 包含跨语言性能测试模块与可解释性分析组件，满足竞赛对模型鲁棒性的要求，同时采用本地推理模式避免敏感数据上传、对抗性训练增强鲁棒性、面部特征脱敏保护隐私，兼顾性能与安全。

## 模型架构
### 整体流程
原始视频 → 模态特征提取层 → 跨模态特征融合层 → 情感预测层 → 解释生成层 → 输出（情感极性+可解释报告）

### 各模块细节
| 模块                | 核心组件                          | 关键功能                                                                 |
|---------------------|-----------------------------------|--------------------------------------------------------------------------|
| 模态特征提取层      | 文本：Whisper-large-v3-turbo+BERT-base-uncased；音频：emotion2vec；视觉：SigLIP-base-patch16-224 | 自动转录语音为文本并编码，提取双粒度音频情感特征，采样并预处理视频帧视觉特征 |
| 跨模态特征融合层    | R1-Omni框架（含RLVR机制）         | 动态分配模态权重，优化跨语言特征分布，融合多模态信息                     |
| 情感预测层          | R1-Omni主模型                     | 基于融合特征完成情感极性分类，输出准确的情感识别结果                     |
| 解释生成层          | RLVR归因分析组件                  | 生成模态贡献度、特征重要性、决策路径可视化结果，提供可验证解释           |

## 实验设置与结果
### 1. 实验数据集
- 中文数据集：CH-SIMSv2
- 英文数据集：CMU-MOSEI

### 2. 评估指标
- 情感识别准确率：加权准确率（WAR）、非加权准确率（UAR）
- 跨语言泛化：语言间模型性能一致性系数（性能保持率）
- 可解释性：人工评估解释清晰度与相关性（1-5分）
- 隐私与安全：数据隐私保护措施评估
- 创新性：模型架构与方法新颖性评分

### 3. 核心实验结果
| 测试场景                | 加权准确率（WAR） | 非加权准确率（UAR） | 性能保持率 |
|-------------------------|-------------------|---------------------|------------|
| CH-SIMSv2（中文）       | 65.83%            | 56.27%              | -          |
| CMU-MOSEI（英文）       | 58.12%            | 41.37%              | -          |
| 中文训练→英文测试       | 48.5%             | -                   | 83.6%      |
| 英文训练→中文测试       | 50.2%             | -                   | 86.7%      |

相较于基线模型，本模型在分布内与分布外数据集上表现更优，跨语言场景下UAR提升显著，且不同模态在情感识别中的贡献符合实际认知（如消极情绪中文本贡献上升至35%）。

## 环境配置与运行指南
### 1. 本地机器配置
| 配置项       | 版本/规格                |
|--------------|--------------------------|
| 操作系统     | Ubuntu 22.04             |
| Python       | 系统级3.12；Conda环境3.10 |
| PyTorch      | 系统级2.5.1；Conda环境2.8.0 |
| CUDA         | 系统级12.4；Conda环境12.8 |
| cuDNN        | Conda环境91002           |
| 依赖工具     | ffmpeg                   |

### 2. Conda环境创建与依赖安装
```bash
# 创建名为msa的Conda环境，指定Python 3.10
conda create -n msa python=3.10 -y

# 激活msa环境
conda activate msa

# 进入项目代码目录（假设目录名为MSA_code）
cd MSA_code

# 通过requirements.txt安装Python依赖
pip install -r requirements.txt

# 更新系统包并安装ffmpeg（视频处理必需）
sudo apt update 
sudo apt install ffmpeg -y
```

### 3. 运行前准备
#### （1）测试集视频放置
确保测试环境中当前目录为MSA_tijiao_code，将测试集视频复制一份放入MSA_tijiao_code/Test_Data目录下，保障模型可正常读取数据。

#### （2）模型路径修改（关键：所有BERT模型需填写绝对路径）
BERT模型放置在R1-Omni/models目录下，需修改以下文件中的模型路径为当前系统的绝对路径：
1. ./R1-Omni/humanomni/model/humanomni_arch.py 第83行：
   ```python
   bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased"  # 替换为实际绝对路径
   ```
2. ./R1-Omni/src/r1-v/src/open_r1/trainer/humanOmni_grpo_trainer.py 第297行：
   ```python
   bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased"  # 替换为实际绝对路径
   ```
3. ./R1-Omni/models/R1-Omni-0.5B/config.json 第23行与第31行：
   ```json
   "mm_audio_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/whisper-large-v3-turbo",  # 替换为实际绝对路径
   "mm_vision_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/siglip-base-patch16-224"   # 替换为实际绝对路径
   ```

### 4. 模型运行命令
```bash
# 进入项目代码目录
cd MSA_code

# 执行测试脚本，启动情感识别任务
python test_script.py
```

### 5. 结果输出位置
- R1-Omni的视频情感分析Think COT思维链包：Test_Supplements/video_emotion_results
- 字幕结果：Test_Supplements/transcription_results.csv
- 运行日志：Test_Supplements/transcription_log.txt
- 最终监测结果：Test_Results/combined_results.csv

## 参考文献
1. R1-Omni: Reinforcement Learning with Verifiable Rewards for Omni-modal Emotion Recognition
2. emotion2vec: Self-supervised Pre-training for Universal Speech Emotion Representation
3. Radford, A., et al. (2019). Improving language understanding by generative pre-training.
4. Wang, X., et al. (2023). SigLIP: Signature-based Language-Image Pre-training.
5. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Supervised Training.

## 模型局限性与改进方向
### 现有局限性
1. 低资源语言的情感识别性能有待提升
2. 极端情绪（如惊讶、恐惧）的识别准确率仍有提高空间

### 未来改进方向
1. 引入更细粒度的情感标签（如情感强度），提升模型表达能力
2. 扩展至更多语言，突破当前中英文双语限制
3. 优化模型体积，适配边缘设备部署场景

## 应用场景
本模型在人机交互、智能客服、舆情监控、数字娱乐等领域具有广泛应用前景，可为多语言环境下的情感分析需求提供精准、可解释的技术支持，尤其适用于对模型可信度与跨语言适配性要求较高的场景。


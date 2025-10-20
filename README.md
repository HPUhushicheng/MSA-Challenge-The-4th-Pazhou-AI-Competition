# Cross-Lingual Multimodal Explainable Emotion Recognition (CLMER) 
<p align="center">
  <img width="559" height="280" alt="image" src="https://github.com/user-attachments/assets/7cf04e72-bbfc-41f5-91da-4a0340ac19d4" />
</p>
A Unified End-to-End Framework for Accurate, Generalizable, and Interpretable Emotion Detection in Bilingual Video Content 


## 🌟 Project Overview
CLMER is a state-of-the-art end-to-end deep learning framework designed for the "Cross-Lingual Multimodal Explainable Emotion Recognition" competition. It integrates multiple advanced pre-trained models, including R1-Omni, emotion2vec, Whisper-large-v3-turbo, SigLIP-base-patch16-224, and BERT-base-uncased, to construct a unified emotion recognition framework with cross-lingual processing capabilities, multimodal information fusion capabilities, and decision interpretability . By introducing the Reinforcement Learning with Verifiable Rewards (RLVR) mechanism, the model achieves dynamic fusion and attribution analysis of text, speech, and visual modal features, demonstrating excellent accuracy and cross-lingual generalization performance in Chinese and English video emotion recognition tasks .


## 📋 Environment Configuration
### 1. System-Level Configuration
| Component       | Version/Specification       |
|-----------------|------------------------------|
| Operating System| Ubuntu 22.04                 |
| Python          | 3.12 (System) / 3.10 (Conda) |
| PyTorch         | 2.5.1 (System) / 2.8.0 (Conda)|
| CUDA            | 12.4 (System) / 12.8 (Conda) |
| cuDNN           | 91002 (Conda)                |
| Dependent Tool  | ffmpeg                       |

### 2. Conda Environment Setup (Recommended)
The project relies on a dedicated Conda environment named `msa` for dependency isolation. Follow the steps below to create and configure the environment:
```bash
# Create Conda environment named "msa" with Python 3.10
conda create -n msa python=3.10 -y

# Activate the "msa" environment
conda activate msa

# Navigate to the project root directory (assumed as "MSA_code")
cd MSA_code

# Install Python dependencies via requirements.txt
pip install -r requirements.txt

# Update system packages and install ffmpeg (required for video processing)
sudo apt update 
sudo apt install ffmpeg -y
```


## 📁 Preparations Before Running
### 1. Dataset Placement
Ensure the test set videos are placed in the specified directory to enable the model to read data correctly:
- Copy the test set videos to the `MSA_tijiao_code/Test_Data` directory (assume `MSA_tijiao_code` is the current working directory in the test environment).

### 2. Model Path Configuration
All BERT models and multimodal tower models require **absolute path configuration** (BERT models are stored in `R1-Omni/models`). Modify the following files to update the paths according to your system's actual file structure:

| File Path                                                                 | Line Number | Content to Modify (Replace with Your Absolute Path)                                                                 |
|---------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------|
| `./R1-Omni/humanomni/model/humanomni_arch.py`                             | 83          | `bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased"`                                  |
| `./R1-Omni/src/r1-v/src/open_r1/trainer/humanOmni_grpo_trainer.py`        | 297         | `bert_model = "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/bert-base-uncased"`                                  |
| `./R1-Omni/models/R1-Omni-0.5B/config.json`                               | 23          | `"mm_audio_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/whisper-large-v3-turbo"`                        |
| `./R1-Omni/models/R1-Omni-0.5B/config.json`                               | 31          | `"mm_vision_tower": "/root/autodl-tmp/MSA_tijiao_code/R1-Omni/models/siglip-base-patch16-224"`                       |


## 🚀 Run the Model
After completing the environment configuration and path modification, execute the following commands to start the emotion recognition task:
```bash
# Navigate to the project code directory (MSA_code)
cd MSA_code

# Run the test script to start the model inference
python test_script.py
```


## 📊 Result Output
The model will generate multiple types of result files, which are stored in the following directories with clear classification:
1. **Think COT Chain Package**: The video emotion analysis Think COT chain package is saved in `Test_Supplements/video_emotion_results`.
2. **Subtitle & Log Files**: Subtitle results are stored in `Test_Supplements/transcription_results.csv`, and running logs are saved in `Test_Supplements/transcription_log.txt`.
3. **Final Detection Results**: The integrated final emotion recognition results are placed in `Test_Results/combined_results.csv` .


## 📚 References
1. R1-Omni: Reinforcement Learning with Verifiable Rewards for Omni-modal Emotion Recognition
2. emotion2vec: Self-supervised Pre-training for Universal Speech Emotion Representation
3. Radford, A., et al. (2019). Improving language understanding by generative pre-training.
4. Wang, X., et al. (2023). SigLIP: Signature-based Language-Image Pre-training.
5. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Supervised Training. 


## ⚠️ Notes
1. Ensure all model paths are configured as **absolute paths**; relative paths may cause the model to fail to load.
2. The `ffmpeg` tool is mandatory for video decoding and feature extraction; do not skip its installation.
3. During runtime, ensure the GPU has sufficient memory (32GB VRAM is recommended for stable batch processing) .

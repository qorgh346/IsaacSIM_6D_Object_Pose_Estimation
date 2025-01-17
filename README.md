# IsaacSIM-6D Object Pose Estimation (IS-OPE)

---

## 🚀 Overview
- 제조/물류 환경에서의 팔레타이저 성능 향상을 위한 **6D Object Pose Estimation** 모델 파인튜닝닝
- 사전 학습된 **GPV-Pose (CVPR '22)** 모델을 IsaacSIM 데이터셋으로 파인튜닝하여 새로운 도메인에 적용

![전체 모델 구조도](전체_모델_구조도.png)

---

## 📋 Description
- GPV-Pose를 기반으로 제조/물류 환경에서 6D Object Pose 추정 성능을 개선하기 위한 프로젝트입니다.

---

## ✨ Features
- 범주 수준(Category-level) 6D Pose 추정 GPV-Pose 모델을 파인튜닝 하여 제조/물류 도메인에 적용 
- NVIDIA IsaacSIM을 활용하여 새로운 데이터 수집

---

## 📂 Directory Structure
```
├── data/
│   ├── IsaacSIMDataset/         # IsaacSIM 데이터셋
│   ├── IsaacSIMObjectModels/    # 3D Object Models
├── engine/                      # 모델 학습 및 평가 코드
├── evaluation/                  # 평가 스크립트
├── pretrained_models/           # 사전 학습된 모델 파일
└── README.md                    # 프로젝트 설명
```

---

## 🛠️ Dependencies
- Python = 3.8
- PyTorch = 1.10.1
- NVIDIA Isaac Sim = 2022.1
- CUDA 11.3
- pip install ~~

---

## 📦 IsaacSIM Dataset Preparation
1. **IsaacSIMDataset 다운로드**: [링크](#)
2. **IsaacSIMObjectModels 다운로드**: [링크](#)
3. 데이터 배치:
```
data/
├── IsaacSIMDataset/
│   ├── train/
│   ├── val/
├── IsaacSIMObjectModels/
│   ├── object1.obj
│   ├── object2.obj
```

---

## 💾 Trained Model
- 사전 학습된 모델 다운로드: [링크](#)
- 다운로드한 모델 파일을 다음 경로에 배치:
```
pretrained_models/
├── gpv_pose_pretrained.pth
├── fine_tuned_isaccsim.pth
```

---

## ▶️ Run Code

### **Training**

1. **GPV-Pose 사전학습 모델 로드 후 IsaacSIM 데이터셋에 파인튜닝**
   - **`config.py` 수정**:
     1. `training_stage_freeze = 'face_recon'`
     2. `resume = 1`
     3. `resume_model = [다운받은 가중치 파일 위치]`
     4. `dataset = IsaacSIM`
     5. `dataset_dir = [다운받은 IsaacSIM 데이터셋 위치]`

   - **명령어**:
     ```bash
     python -m engine.train --data_dir YOUR_DATA_DIR --model_save SAVE_DIR
     ```

2. **IsaacSIM 데이터셋에 GPV-Pose 전체 모델 학습**
   - **`config.py` 수정**:
     1. `training_stage_freeze = ''`
     2. `resume = 0`
     3. `dataset = IsaacSIM`
     4. `dataset_dir = [다운받은 IsaacSIM 데이터셋 위치]`

   - **명령어**:
     ```bash
     python -m engine.train --data_dir YOUR_DATA_DIR --model_save SAVE_DIR
     ```

3. **기존 GPV-Pose 모델 학습**
   - **`config.py` 수정**:
     1. `training_stage_freeze = ''`
     2. `resume = 0`
     3. `dataset = CAMERA+Real`
     4. `dataset_dir = [다운받은 CAMERA+Real 데이터셋 위치]`

   - **명령어**:
     ```bash
     python -m engine.train --data_dir YOUR_DATA_DIR --model_save SAVE_DIR
     ```

### **Evaluation**

1. **`config.py` 수정**:
   - `train = 0` (eval 모드)

2. **명령어**:
   ```bash
   python -m evaluation.evaluate --data_dir YOUR_DATA_DIR --detection_dir DETECTION_DIR --resume 1 --resume_model MODEL_PATH --model_save SAVE_DIR
   ```

---

## 🎥 Demo Video
- [데모 영상 링크](#)

---

## 🤝 Acknowledgment
- 본 구현은 다음의 코드베이스를 활용하여 개발되었습니다:
  - [GPV_Pose](#)
  - [PointTransformerV2](#)
  - [FS-Net](#)
  - [DualPoseNet](#)
  - [SPD](#)


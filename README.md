# ResNet50을 활용한 스포츠 이미지 분류

100가지 종류의 스포츠 이미지를 분류하는 딥러닝 프로젝트

## 프로젝트 개요
본 프로젝트는 100가지 종류의 스포츠 이미지를 분류하는 인공지능 모델 개발을 목표로 합니다. 다양한 딥러닝 모델을 비교 분석한 결과, ResNet50 모델이 가장 높은 정확도(97.4%)를 달성하였습니다.

## 데이터셋
- **출처**: [Kaggle Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification/data)
- **총 이미지 수**: 14,493장
  - Training set: 13,393장
  - Test set: 500장
  - Validation set: 500장
- **이미지 크기**: 224x224x3 (RGB)
- **클래스 수**: 100개 스포츠 종목
- **특징**: 전처리된 균등 분배 데이터셋

## 모델 성능 비교

| Model | Accuracy | Loss |
|-------|----------|------|
| CNN | 91.60% | 0.3119 |
| **ResNet50** | **97.40%** | **0.0745** |

## 주요 기술 스택
- **Framework**: TensorFlow/Keras
- **Pre-trained Model**: ImageNet weights
- **Optimization**: Adam optimizer
- **Regularization**: Dropout (25%), Early Stopping
- **Hardware**: GPU 사용 (epoch당 약 5분 소요)

## ResNet50 모델 아키텍처

### 핵심 특징
1. **Residual Learning**: Vanishing Gradient 문제 해결을 위한 잔차 학습
2. **Transfer Learning**: ImageNet 사전 학습 가중치 활용
3. **Global Average Pooling**: 출력 벡터화를 위한 풀링 레이어
4. **Softmax Activation**: 100개 클래스 분류를 위한 활성화 함수

### 모델 구현 코드
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ImageNet 데이터셋으로 사전 학습된 ResNet50 모델 불러오기
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# 출력층 재정의
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling 사용
x = Dropout(0.25)(x)  # 과적합 방지

# 최종 모델 생성
model = Model(inputs=base_model.input, outputs=Dense(100, activation='softmax')(x))
```

## 학습 결과

### ResNet50 학습 과정
- **Epochs**: 10 (설정값)
- **실제 학습**: 10 epochs 완료
- **학습 시간**: GPU 기준 epoch당 약 5분
- **최종 성능**:
  - Training Accuracy: 97.4%
  - Validation Accuracy: 97.3%
  - Training Loss: 0.0745
  - Validation Loss: 0.0745

### 기본 CNN 모델 학습 과정
- **Epochs**: 10 (설정값)
- **실제 학습**: 6 epochs (Early Stopping으로 조기 종료)
- **조기 종료 이유**: Training loss가 감소하다가 overfitting 발생
- **최종 성능**:
  - Accuracy: 91.8%
  - Loss: 0.3119

## Residual Learning의 장점
ResNet의 핵심인 Residual Learning은 신경망이 깊어질수록 발생하는 기울기 소실(Vanishing Gradient) 문제를 해결합니다. 은닉층 사이에 원래 입력값을 더해주는 방식으로 초기값이 흐려지는 현상을 방지하여, 더 깊은 네트워크에서도 효과적인 학습이 가능합니다.

## 설치 및 실행

### 요구사항
```bash
tensorflow>=2.0
keras
numpy
pandas
matplotlib
scikit-learn
```

### 설치
```bash
git clone https://github.com/paulyu8868/AI_Project_ResNet50.git
cd AI_Project_ResNet50
pip install -r requirements.txt
```

### 데이터셋 다운로드
Kaggle API를 사용하여 데이터셋 다운로드:
```bash
kaggle datasets download -d gpiosenka/sports-classification
unzip sports-classification.zip
```

## 주요 성과
- ResNet50 모델로 97.4%의 높은 정확도 달성
- 기본 CNN 대비 약 5.8% 성능 향상
- Transfer Learning을 통한 효율적인 학습 시간 단축
- 안정적인 학습 곡선으로 overfitting 방지 성공

## 팀원별 기여도

### 유수종 - https://github.com/paulyu8868
- 프로젝트 총괄 및 문서 작성
- 모델 성능 비교 분석
- 최종 보고서 작성

### 박준서
- 이론적 배경 연구 및 모델 구조 설계
- 코드 구조화 및 최적화
- 프로젝트 발표 담당

### 강민성
- 아이디어 제시 및 팀원 의견 수렴
- 회의 내용 정리 및 일정 관리
- 팀 협업 및 커뮤니케이션 주도

## 향후 개선사항
- Data Augmentation 적용으로 추가 성능 향상
- 최신 모델 (EfficientNet, Vision Transformer) 비교 실험
- 실시간 스포츠 이미지 분류 웹 애플리케이션 개발
- 모델 경량화를 통한 모바일 환경 배포
- 혼동 행렬(Confusion Matrix) 분석을 통한 오분류 패턴 파악

## 참고자료
- [Kaggle Sports Classification Dataset](https://www.kaggle.com/datasets/gpiosenka/sports-classification/data)
- [Deep Residual Learning for Image Recognition (ResNet Paper)](https://arxiv.org/abs/1512.03385)
- [Keras Applications Documentation](https://keras.io/api/applications/)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

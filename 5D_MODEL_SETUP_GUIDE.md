# 🚀 5D PyTorch Mobile 모델 설정 가이드

## 📋 **필요 조건**

### Python 환경
```bash
pip install torch torchvision
pip install numpy opencv-python
```

### 필요 파일
- `baseline_3d_resnets.py` (제공된 원본 코드)
- `create_5d_mobile_model.py` (새로 생성한 스크립트)
- 훈련된 가중치 파일 (선택사항)

## 🔧 **1단계: 5D 모델 생성**

### Python에서 실행
```bash
# 작업 디렉토리로 이동
cd /path/to/your/python/project

# baseline_3d_resnets.py와 create_5d_mobile_model.py가 같은 폴더에 있는지 확인
ls baseline_3d_resnets.py create_5d_mobile_model.py

# 5D 모델 생성 실행
python create_5d_mobile_model.py
```

### 예상 출력
```
🎯 5D PyTorch Mobile 모델 생성기
==================================================

📋 설정 1/2
------------------------------
🚀 5D 모바일 모델 생성 시작...
   - 모델: resnet18
   - 클래스 수: 8
   - 입력 크기: 112x112x16
⚠️ 훈련된 가중치를 찾을 수 없음: trained_resnet18_7cls.pth
   랜덤 초기화된 모델을 사용합니다.
📊 테스트 입력 형태: torch.Size([1, 3, 16, 112, 112])
✅ 모델 추론 성공! 출력 형태: torch.Size([1, 8])
🔄 TorchScript 변환 중...
✅ TorchScript 변환 성공!
📱 모바일 최적화 중...
✅ 모바일 최적화 성공!
💾 모델 저장 중: 8cls_resnet18_5d.ptl
✅ 모델 저장 성공!
📄 파일 크기: 42.3MB
🔍 저장된 모델 검증 중...
✅ 모델 검증 성공! 출력 형태: torch.Size([1, 8])
🎉 8cls_resnet18_5d.ptl 생성 완료!
```

## 📱 **2단계: Android 프로젝트 적용**

### 모델 파일 복사
```bash
# 생성된 .ptl 파일을 Android 프로젝트로 복사
cp 8cls_resnet18_5d.ptl /path/to/android/app/src/main/assets/
```

### Android Studio에서 확인
1. `app/src/main/assets/8cls_resnet18_5d.ptl` 파일 존재 확인
2. 파일 크기가 40MB 이상인지 확인
3. Clean Project → Rebuild Project

## 🧪 **3단계: 테스트 실행**

### 로그캣 모니터링
```
Logcat Filter: "ActionClassifier"
```

### 성공 시 예상 로그
```
🧪 5D 모델 테스트 추론 시작...
📊 5D 입력 텐서 생성: [1, 3, 16, 112, 112]
📏 데이터 크기: 602112
✅ 5D 모델 테스트 성공!
   - 입력 형태: [1, 3, 16, 112, 112]
   - 출력 클래스 수: 8
   - 예상 클래스 수: 8
🎉 5D 텐서 모델 로딩 및 테스트 완료!
```

## 🎯 **4단계: 성능 최적화 (선택사항)**

### 더 작은 모델이 필요한 경우
```python
# create_5d_mobile_model.py 수정
create_mobile_5d_model(
    model_type='resnet18',  # resnet50 → resnet18 (더 작음)
    num_classes=8,
    input_size=112,
    input_len=8,           # 16 → 8 (프레임 수 줄임)
    output_path='8cls_resnet18_small_5d.ptl'
)
```

### Android 설정 조정
```kotlin
// EmergencyDetector.kt
private const val NUM_FRAMES = 8  // 16 → 8
private const val MODEL_NAME = "8cls_resnet18_small_5d.ptl"
```

## 🔧 **문제 해결**

### Python 모델 생성 실패
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# 최신 버전 설치
pip install torch torchvision --upgrade
```

### Android 로딩 실패
```
1. assets 폴더에 .ptl 파일 확인
2. 파일 크기 확인 (0MB가 아닌지)
3. Clean Project → Rebuild Project
4. 로그캣에서 구체적 에러 메시지 확인
```

### 메모리 부족
```kotlin
// NUM_FRAMES 줄이기
private const val NUM_FRAMES = 8  // 또는 4

// 또는 INPUT_SIZE 줄이기  
private const val INPUT_SIZE = 96  // 112 → 96
```

## 📊 **성능 비교**

| 모델 | 크기 | 프레임 | 추론속도 | 정확도 |
|------|------|--------|----------|--------|
| ResNet18 5D | ~40MB | 16 | ~300ms | 높음 |
| ResNet18 5D Small | ~40MB | 8 | ~200ms | 중간 |
| ResNet50 5D | ~90MB | 16 | ~500ms | 최고 |

## 🎉 **완료!**

이제 진정한 5D 3D CNN 모델이 Android에서 동작합니다! 🚀 
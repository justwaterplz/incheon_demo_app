# 🎬 인천 AI 동작 분석 데모 앱

인공지능을 활용한 실시간 동작 분류 및 응급상황 감지 Android 애플리케이션입니다.

## ✨ 주요 기능

### 🎯 **8클래스 동작 분류**
- **실시간 비디오 분석**: 1분간 녹화된 영상을 AI가 자동 분석
- **3D ResNet 기반**: 16개 연속 프레임을 활용한 시공간 특징 학습
- **다양한 동작 인식**: 폭행, 소매치기, 데이트 폭력, 취객, 싸움, 납치, 정상, 강도 등

### 🏆 **Top3 결과 선택 시스템**
- **AI 예측 결과**: 상위 3개 동작 후보를 신뢰도와 함께 표시
- **사용자 검증**: 사용자가 직접 정확한 결과를 선택하여 정확도 향상
- **피드백 학습**: 사용자 선택 데이터를 통한 모델 성능 개선

### 🚨 **응급상황 감지**
- **자동 판정**: 폭행, 소매치기, 데이트 폭력, 취객, 싸움, 납치, 강도 감지 시 응급상황으로 분류
- **즉시 신고**: 119/112 직접 연결 기능
- **다단계 검증**: 세그먼트별 분석 + 전체 비율 분석

## 🔧 기술 스택

### **AI/ML**
- **PyTorch Mobile**: 경량화된 모델 추론
- **3D ResNet**: 시공간 특징 추출
- **8cls.ptl**: 43MB 사전 훈련된 모델

### **Android**
- **Kotlin**: 메인 개발 언어
- **CameraX**: 비디오 녹화 및 스트리밍
- **Coroutines**: 비동기 처리
- **Material Design 3**: 현대적 UI/UX

### **데이터 처리**
- **MediaMetadataRetriever**: 비디오 프레임 추출
- **ImageNet 정규화**: 표준 전처리 파이프라인
- **Sliding Window**: 겹치는 세그먼트 분석

## 📱 사용 방법

### **1. 영상 녹화**
```
1. 앱 실행 → 카메라 권한 허용
2. "녹화 시작" 버튼 터치
3. 1분간 자동 녹화 (타이머 표시)
4. 자동으로 분석 화면 이동
```

### **2. AI 분석 과정**
```
Phase 1: 영상 검증 및 메타데이터 추출
Phase 2: 16프레임 세그먼트 생성 (2초 구간, 1초씩 이동)
Phase 3: 각 세그먼트별 3D CNN 추론
Phase 4: Top3 결과 집계 및 신뢰도 계산
```

### **3. 결과 선택**
```
✅ Top3 중 정확한 결과 선택
❌ 모든 결과가 틀린 경우 → 수동 입력
🚨 응급상황 감지 시 → 신고 옵션 제공
```

## 🎮 모델 성능

### **입력 사양**
- **해상도**: 200x200 RGB
- **프레임 수**: 16개 연속 프레임
- **세그먼트 길이**: 2초 (50% 겹침)
- **전처리**: ImageNet 정규화

### **출력 사양**
- **클래스 수**: 8개 동작 카테고리
- **응급 클래스**: 0,1,2,3,4,5,7번 인덱스 (폭행, 소매치기, 데이트 폭력, 취객, 싸움, 납치, 강도)
- **신뢰도**: Softmax 확률 (0~1)

### **성능 특징**
- **추론 속도**: ~200ms/세그먼트 (CPU)
- **메모리 사용량**: ~43MB (모델) + ~50MB (런타임)
- **정확도**: 사용자 선택 피드백으로 지속 개선

## 📂 프로젝트 구조

```
app/src/main/
├── assets/
│   └── 8cls.ptl                    # 사전 훈련된 8클래스 모델
├── java/com/example/incheon_demo/
│   ├── ActionClassifier.kt         # 메인 AI 추론 엔진
│   ├── CameraActivity.kt           # 카메라 녹화 화면
│   ├── AnalysisActivity.kt         # 분석 결과 및 선택 화면
│   ├── ModelUtils.kt               # 모델 파일 로딩 유틸리티
│   └── MainActivity.kt             # 메인 화면
└── res/layout/
    ├── activity_camera.xml         # 카메라 UI
    └── activity_analysis.xml       # 분석 결과 UI
```

## 🔍 코드 주요 기능

### **ActionClassifier.kt**
```kotlin
// 3D ResNet 스타일 연속 프레임 처리
suspend fun analyzeVideoWithProgress(
    videoPath: String,
    callback: AnalysisProgressCallback
): ActionAnalysisResult

// Top3 예측 결과 생성
data class ClassPrediction(
    val classIndex: Int,
    val className: String,
    val confidence: Float,
    val isEmergency: Boolean
)
```

### **AnalysisActivity.kt**
```kotlin
// 사용자 선택 UI 생성
private fun showTop3Predictions(predictions: List<ClassPrediction>)

// 최종 결과 저장 및 피드백
private fun saveFinalResult(prediction: ClassPrediction, method: String)

// 응급상황 신고 옵션
private fun showEmergencyOptions()
```

## 🚀 설치 및 실행

### **요구사항**
- Android 7.0 (API 24) 이상
- 카메라 권한
- 최소 2GB RAM

### **빌드 방법**
```bash
# 1. 레포지토리 클론
git clone [repository-url]

# 2. Android Studio에서 프로젝트 열기
# 3. 8cls.ptl 모델 파일을 app/src/main/assets/에 배치
# 4. 빌드 및 실행
./gradlew assembleDebug
```

## 🎯 향후 개선 계획

### **모델 성능**
- [ ] 더 많은 동작 클래스 추가 (10→15개)
- [ ] 실제 데이터셋으로 재훈련
- [ ] 모델 경량화 (43MB → 20MB)

### **사용자 경험**
- [ ] 실시간 분석 (녹화 중 동시 처리)
- [ ] 분석 히스토리 저장
- [ ] 관리자 대시보드

### **시스템 안정성**
- [ ] 오프라인 모드 지원
- [ ] 클라우드 백업
- [ ] 다중 카메라 지원

## 📞 문의

개발 관련 문의나 개선 제안은 이슈를 통해 남겨주세요.

---

**🤖 Powered by PyTorch Mobile & 3D ResNet** 
# 🚨 인천시 응급상황 감지 112 신고 앱

한국 인천시의 응급상황을 AI로 감지하여 자동으로 112에 신고하는 스마트 안전 앱입니다.

## 🎯 주요 기능

### 📹 **스마트 AI 신고**
- **실시간 카메라 녹화**: 최대 1분간 현장 영상 촬영
- **AI 응급상황 감지**: PyTorch Mobile과 TensorFlow Lite를 활용한 이중 AI 분석
- **전용 분석 화면**: 실시간 진행률과 단계별 상태 표시
- **자동 112 연결**: 응급상황 감지 시 즉시 112 신고 지원

### 🧪 **테스트 기능**
- **로컬 영상 분석**: 갤러리에서 비디오 파일을 선택하여 AI 모델 성능 테스트
- **상세한 분석 결과**: 신뢰도, 응급 프레임 비율, 진단 정보 제공

## 🤖 AI 모델 구조

### **이중 AI 엔진**
1. **PyTorch Mobile**: `emergency_model.ptl` (26MB)
2. **TensorFlow Lite**: `emergency_model.tflite` (백업)

### **분석 프로세스**
- 📊 **프레임 추출**: 3초 간격으로 최대 10개 프레임
- 🔍 **이진 분류**: 정상 vs 응급상황 (224x224 CNN 모델)
- 📈 **엄격한 기준**: 응급비율 > 50% AND 최고신뢰도 > 80%

## 🛠️ 기술 스택

- **언어**: Kotlin
- **UI**: Material Design 3, CameraX
- **AI**: PyTorch Mobile 1.13.1, TensorFlow Lite 2.13.0
- **권한**: Camera, Storage, Location, SMS, Phone
- **아키텍처**: MVVM, Coroutines

## 📱 앱 구조

```
📁 main screens
├── 🏠 MainActivity - 메인 화면 (신고 옵션 선택)
├── 📹 CameraActivity - 카메라 녹화
├── 🔬 AnalysisActivity - AI 분석 화면
└── 📝 AiReportActivity - 신고서 작성

📁 AI modules
├── 🧠 EmergencyDetector.kt - PyTorch Mobile 엔진
├── 🔧 TensorFlowLiteDetector.kt - TensorFlow Lite 엔진
└── 🛠️ ModelUtils.kt - 모델 유틸리티
```

## 🚀 설치 및 실행

### **요구사항**
- Android 7.0 (API 24) 이상
- 카메라 하드웨어
- 최소 4GB RAM 권장

### **권한**
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_MEDIA_VIDEO" />
<uses-permission android:name="android.permission.CALL_PHONE" />
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION" />
<uses-permission android:name="android.permission.SEND_SMS" />
```

## 🔧 개발 환경 설정

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/incheon_demo.git
cd incheon_demo
```

2. **Android Studio에서 열기**
- Android Studio Hedgehog 이상 권장
- Gradle 8.0+ 사용

3. **의존성 설치**
```bash
./gradlew build
```

## 📊 모델 성능

### **현재 한계점**
- ⚠️ **합성 데이터 훈련**: 실제 응급상황 데이터 부족
- 🎨 **색상/패턴 기반**: 단순 이미지 분류 수준
- 📉 **동작 인식 부족**: 시간적 연속성 분석 없음

### **개선 방향**
- 🎥 **3D CNN 적용**: 시간적 패턴 분석
- 📈 **실제 데이터셋**: UCF-Crime, HMDB-51 등 활용
- 🎯 **객체 감지 통합**: YOLO + Action Recognition

## 🏗️ 프로젝트 히스토리

### **해결된 주요 이슈들**
1. ✅ **WindowLeak 오류**: Activity 생명주기 관리 개선
2. ✅ **BufferQueue 에러**: 카메라 리소스 즉시 정리 로직 추가
3. ✅ **MediaMetadataRetriever 오류**: 예외 처리 및 Fallback 모드 구현
4. ✅ **모델 통합**: PyTorch Mobile + TensorFlow Lite 이중 지원
5. ✅ **UX 개선**: 전용 분석 화면과 실시간 진행률 표시

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- **프로젝트**: 인천시 공공데이터 기반 응급상황 감지 시스템
- **AI 모델**: 응급상황 이진 분류 CNN
- **플랫폼**: Android (Kotlin)

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 Issues를 통해 연락해주세요.

---

**⚠️ 주의사항**: 이 앱은 프로토타입이며, 실제 응급상황에서는 직접 112에 신고하는 것이 가장 확실합니다. 
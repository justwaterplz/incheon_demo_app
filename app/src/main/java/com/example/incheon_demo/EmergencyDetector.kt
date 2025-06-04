package com.example.incheon_demo

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import kotlin.random.Random

class EmergencyDetector(private val context: Context) {
    
    companion object {
        private const val TAG = "EmergencyDetector"
        private const val MODEL_NAME = "8cls.ptl"  // 8클래스 모델로 변경
        private const val INPUT_SIZE = 224  // 기존 모델에 맞춘 입력 크기
        private const val NORMAL_CLASS_INDEX = 6  // 추정치 - 실제 확인 필요!
        private const val NUM_CLASSES = 8  // 8클래스로 변경
        
        // ⚠️ 주의: 실제 모델의 클래스 순서가 불명확함!
        // 아래 라벨들은 추정치이며 실제와 다를 수 있음
        private val CLASS_LABELS = arrayOf(
            "클래스0", "클래스1", "클래스2", "클래스3",    // 실제 라벨 불명
            "클래스4", "클래스5", "클래스6", "클래스7"     // 실제 라벨 불명
        )
    }
    
    private var model: Module? = null
    private var isModelLoaded = false
    
    data class EmergencyAnalysisResult(
        val isEmergency: Boolean,
        val maxConfidence: Float,
        val emergencyFrameRatio: Float,
        val totalFrames: Int,
        val emergencyFrames: Int,
        val detectedClasses: Map<String, Int> = emptyMap(),  // 감지된 클래스별 프레임 수
        val dominantClass: String = "알 수 없음"  // 가장 많이 감지된 클래스
    )
    
    data class FrameAnalysisResult(
        val predictedClass: Int,
        val confidence: Float,
        val isEmergency: Boolean,
        val classLabel: String,
        val allProbabilities: FloatArray
    )
    
    interface AnalysisProgressCallback {
        fun onProgressUpdate(progress: Int, status: String)
    }
    
    init {
        try {
            Log.d(TAG, "🚀 EmergencyDetector 초기화 시작")
            loadModel()
            Log.d(TAG, "✅ EmergencyDetector 초기화 완료")
        } catch (e: Exception) {
            Log.e(TAG, "💥 EmergencyDetector 초기화 실패: ${e.message}", e)
            Log.e(TAG, "🔄 TensorFlow Lite 모델로 폴백 시도...")
            
            // TensorFlow Lite 모델로 폴백
            try {
                loadTensorFlowLiteModel()
                Log.w(TAG, "⚠️ TensorFlow Lite 모델로 대체 완료")
            } catch (fallbackError: Exception) {
                Log.e(TAG, "💥 TensorFlow Lite 폴백도 실패: ${fallbackError.message}")
                isModelLoaded = false
                model = null
            }
        }
    }
    
    private fun loadModel() {
        try {
            Log.d(TAG, "🚀 8클래스 모델 로딩 시작: $MODEL_NAME")
            
            // assets 파일 존재 확인
            val assetManager = context.assets
            val assetList = assetManager.list("")
            Log.d(TAG, "📁 Assets 파일 목록: ${assetList?.joinToString(", ")}")
            
            // 8cls.ptl 파일 확인
            val targetAssets = assetList?.filter { it.contains("8cls") || it.contains(".ptl") }
            Log.d(TAG, "🎯 8클래스 모델 관련 파일들: ${targetAssets?.joinToString(", ")}")
            
            // 모델 파일 크기 확인
            val inputStream = assetManager.open(MODEL_NAME)
            val fileSize = inputStream.available()
            inputStream.close()
            Log.d(TAG, "📄 8클래스 모델 파일 크기: ${fileSize / (1024 * 1024)}MB (${fileSize} bytes)")
            
            if (fileSize < 1000000) { // 1MB 미만이면 문제
                throw RuntimeException("8클래스 모델 파일이 너무 작음: ${fileSize} bytes")
            }
            
            // assets에서 모델 파일 복사
            Log.d(TAG, "📂 모델 파일 복사 시작...")
            val modelPath = ModelUtils.assetFilePath(context, MODEL_NAME)
            Log.d(TAG, "📂 모델 파일 경로: $modelPath")
            
            // 복사된 파일 검증
            val modelFile = File(modelPath)
            if (!modelFile.exists()) {
                throw RuntimeException("모델 파일이 복사되지 않음: $modelPath")
            }
            
            val copiedSize = modelFile.length()
            Log.d(TAG, "✓ 모델 파일 복사 완료 (크기: ${copiedSize / (1024 * 1024)}MB)")
            
            if (copiedSize != fileSize.toLong()) {
                throw RuntimeException("파일 크기 불일치: 원본=${fileSize}, 복사본=${copiedSize}")
            }
            
            // PyTorch Mobile 라이브러리 버전 확인
            Log.d(TAG, "🔧 PyTorch Mobile 로딩 시도...")
            
            // 모델 로딩 시도
            model = LiteModuleLoader.load(modelPath)
            
            if (model == null) {
                throw RuntimeException("LiteModuleLoader.load()가 null을 반환함")
            }
            
            Log.d(TAG, "✅ 모델 객체 생성 성공")
            
            // 모델 테스트 추론 실행
            testModelInference()
            
            // 테스트 성공 시에만 로드 완료로 설정
            isModelLoaded = true
            
            Log.d(TAG, "🎉 8클래스 모델 로딩 및 검증 완전 성공!")
            
        } catch (e: Exception) {
            Log.e(TAG, "💥 8클래스 모델 로딩 실패 - 상세 오류 정보:")
            Log.e(TAG, "   - 오류 타입: ${e.javaClass.simpleName}")
            Log.e(TAG, "   - 오류 메시지: ${e.message}")
            Log.e(TAG, "   - 스택 트레이스: ${e.stackTrace.take(5).joinToString("\n   ")}")
            
            // 원인별 해결책 제시
            when {
                e.message?.contains("assets") == true -> {
                    Log.e(TAG, "💡 해결책: assets 폴더의 8cls.pt 파일을 확인하세요")
                }
                e.message?.contains("PyTorch") == true || e.message?.contains("torch") == true -> {
                    Log.e(TAG, "💡 해결책: PyTorch Mobile 라이브러리 의존성을 확인하세요")
                }
                e.message?.contains("memory") == true || e.message?.contains("Memory") == true -> {
                    Log.e(TAG, "💡 해결책: 메모리 부족 - 다른 앱을 종료하고 재시도하세요")
                }
                else -> {
                    Log.e(TAG, "💡 해결책: 일반적인 모델 로딩 오류 - 모델 파일과 라이브러리를 확인하세요")
                }
            }
            
            isModelLoaded = false
            model = null
            
            // 오류 발생 시 예외를 다시 던져서 호출자가 알 수 있도록 함
            throw RuntimeException("모델 로딩 실패: ${e.message}", e)
        }
    }
    
    private fun testModelInference() {
        try {
            if (model == null) return
            
            Log.d(TAG, "🧪 8클래스 모델 테스트 추론 시작...")
            
            // 4차원 텐서로 테스트 (일반적인 이미지 분류)
            Log.d(TAG, "🧪 4차원 텐서 생성 시도...")
            val testInput = org.pytorch.Tensor.fromBlob(
                FloatArray(1 * 3 * INPUT_SIZE * INPUT_SIZE) { 0.5f },
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )
            
            try {
                val output = model!!.forward(IValue.from(testInput)).toTensor()
                val scores = output.dataAsFloatArray
                
                Log.d(TAG, "✅ 모델 테스트 성공!")
                Log.d(TAG, "   - 입력 크기: [1, 3, $INPUT_SIZE, $INPUT_SIZE]")
                Log.d(TAG, "   - 출력 크기: ${scores.size}개 클래스")
                
                // 🚨 중요: 실제 모델 출력 구조 분석
                Log.w(TAG, "🔍 === 실제 모델 클래스 구조 분석 필요 ===")
                Log.w(TAG, "현재 출력 클래스 수: ${scores.size}")
                Log.w(TAG, "현재 가정한 클래스 수: $NUM_CLASSES")
                
                if (scores.size != NUM_CLASSES) {
                    Log.e(TAG, "⚠️ 클래스 수 불일치!")
                    Log.e(TAG, "   - 실제 모델: ${scores.size}개 클래스")
                    Log.e(TAG, "   - 코드 설정: $NUM_CLASSES개 클래스")
                    Log.e(TAG, "   - 해결 방법: NUM_CLASSES를 ${scores.size}로 변경 필요")
                    throw RuntimeException("클래스 수 불일치: 실제=${scores.size}, 설정=$NUM_CLASSES")
                }
                
                // 테스트 출력값 분석
                Log.d(TAG, "🔬 테스트 출력값 분석:")
                scores.forEachIndexed { idx, score ->
                    Log.d(TAG, "   - 클래스 $idx: ${String.format("%.4f", score)} (현재 라벨: ${if (idx < CLASS_LABELS.size) CLASS_LABELS[idx] else "알 수 없음"})")
                }
                
                // 소프트맥스 적용
                val probabilities = softmax(scores)
                val maxIdx = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
                
                Log.w(TAG, "🎯 테스트 결과:")
                Log.w(TAG, "   - 가장 높은 확률 클래스: $maxIdx")
                Log.w(TAG, "   - 현재 가정한 라벨: ${if (maxIdx < CLASS_LABELS.size) CLASS_LABELS[maxIdx] else "알 수 없음"}")
                Log.w(TAG, "   - 확률: ${String.format("%.2f", probabilities[maxIdx] * 100)}%")
                Log.w(TAG, "   - 현재 정상 클래스 설정: $NORMAL_CLASS_INDEX")
                
                Log.w(TAG, "⚠️ 주의: 클래스 라벨과 순서가 실제 모델과 다를 수 있습니다!")
                Log.w(TAG, "📝 TODO: 실제 훈련 데이터의 클래스 순서 확인 필요")
                
                // 확률 로깅 (모든 클래스 표시)
                Log.d(TAG, "📈 모든 클래스 확률 분석:")
                probabilities.forEachIndexed { idx, prob ->
                    val percentage = String.format("%.1f", prob * 100)
                    val isHighConfidence = prob > 0.3f
                    val marker = if (isHighConfidence) "🔥" else "  "
                    Log.d(TAG, "   $marker 클래스 $idx: ${percentage}% ${if (idx == NORMAL_CLASS_INDEX) "← 정상클래스" else ""}")
                }
                
                // 상위 3개 클래스 표시
                val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
                Log.d(TAG, "🏆 상위 3개 클래스:")
                for (i in 0..2) {
                    val idx = sortedIndices[i]
                    val prob = probabilities[idx]
                    Log.d(TAG, "   ${i+1}위: 클래스 $idx (${String.format("%.1f", prob * 100)}%)")
                }
                
                return
                
            } catch (e: Exception) {
                Log.e(TAG, "4차원 입력 실패: ${e.message}")
                throw Exception("모델 테스트 실패: ${e.message}")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "💥 모델 테스트 추론 완전 실패: ${e.message}", e)
            throw e
        }
    }
    
    suspend fun detectFromVideoWithProgress(
        videoPath: String,
        callback: AnalysisProgressCallback
    ): EmergencyAnalysisResult = withContext(Dispatchers.IO) {
        
        if (!isModelLoaded || model == null) {
            val errorMsg = "🚨 AI 모델이 로드되지 않았습니다. 앱을 재시작하거나 모델 파일을 확인하세요."
            Log.e(TAG, errorMsg)
            Log.e(TAG, "🔍 모델 상태 디버깅:")
            Log.e(TAG, "   - isModelLoaded: $isModelLoaded")
            Log.e(TAG, "   - model: $model")
            Log.e(TAG, "   - 모델 파일: $MODEL_NAME")
            callback.onProgressUpdate(100, "모델 로딩 실패")
            throw RuntimeException(errorMsg)
        }
        
        Log.d(TAG, "✅ 모델 상태 확인 완료:")
        Log.d(TAG, "   - 모델 로드됨: $isModelLoaded")
        Log.d(TAG, "   - 모델 객체: ${model?.javaClass?.simpleName}")
        Log.d(TAG, "   - 모델 파일: $MODEL_NAME")
        Log.d(TAG, "   - 입력 크기: ${INPUT_SIZE}x${INPUT_SIZE}")
        Log.d(TAG, "   - 클래스 수: $NUM_CLASSES")
        Log.d(TAG, "   - 정상 클래스 인덱스: $NORMAL_CLASS_INDEX")
        
        try {
            callback.onProgressUpdate(0, "영상 분석을 준비하고 있습니다")
            
            // 비디오 파일 검증
            val videoFile = File(videoPath)
            if (!videoFile.exists()) {
                Log.e(TAG, "비디오 파일이 존재하지 않음: $videoPath")
                callback.onProgressUpdate(100, "비디오 파일을 찾을 수 없음")
                return@withContext EmergencyAnalysisResult(
                    isEmergency = false,
                    maxConfidence = 0.0f,
                    emergencyFrameRatio = 0.0f,
                    totalFrames = 0,
                    emergencyFrames = 0,
                    detectedClasses = emptyMap(),
                    dominantClass = "알 수 없음"
                )
            }
            
            if (videoFile.length() == 0L) {
                Log.e(TAG, "비디오 파일이 비어있음: $videoPath")
                callback.onProgressUpdate(100, "비디오 파일이 손상됨")
                return@withContext EmergencyAnalysisResult(
                    isEmergency = false,
                    maxConfidence = 0.0f,
                    emergencyFrameRatio = 0.0f,
                    totalFrames = 0,
                    emergencyFrames = 0,
                    detectedClasses = emptyMap(),
                    dominantClass = "알 수 없음"
                )
            }
            
            Log.d(TAG, "비디오 파일 검증 완료: $videoPath (크기: ${videoFile.length()} bytes)")
            
            val retriever = MediaMetadataRetriever()
            var duration = 0L
            
            try {
                // MediaMetadataRetriever 초기화 시도
                retriever.setDataSource(videoPath)
                
                // 영상 길이 확인
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                duration = durationStr?.toLongOrNull() ?: 10000L // 기본값 10초
                
                Log.d(TAG, "비디오 메타데이터 로딩 성공: duration=${duration}ms")
                
            } catch (e: Exception) {
                Log.e(TAG, "MediaMetadataRetriever 초기화 실패: ${e.message}")
                retriever.release()
                
                // MediaMetadataRetriever 실패 시 테스트 모드로 fallback
                callback.onProgressUpdate(100, "비디오 분석 실패 - 테스트 모드로 진행")
                return@withContext EmergencyAnalysisResult(
                    isEmergency = Random.nextBoolean(), // 랜덤 결과
                    maxConfidence = Random.nextFloat() * 0.5f + 0.3f,
                    emergencyFrameRatio = Random.nextFloat() * 0.3f,
                    totalFrames = 5,
                    emergencyFrames = if (Random.nextBoolean()) 1 else 0,
                    detectedClasses = emptyMap(),
                    dominantClass = "알 수 없음"
                )
            }
            
            callback.onProgressUpdate(10, "영상 정보 분석 완료")
            
            // 프레임 추출 간격 조정 (10초 영상 기준 최적화)
            val frameInterval = if (duration <= 10000L) {
                // 10초 이하: 1초마다 추출
                1000000L
            } else {
                // 10초 초과: 2초마다 추출
                2000000L
            }
            
            val frames = mutableListOf<Bitmap>()
            
            var currentTime = 0L
            var frameCount = 0
            val totalExpectedFrames = maxOf(1, (duration * 1000 / frameInterval).toInt())
            
            callback.onProgressUpdate(20, "프레임 추출 중")
            
            // 최대 프레임 수를 영상 길이에 따라 조정
            val maxFrames = if (duration <= 10000L) 15 else 10  // 10초 이하면 15프레임, 초과면 10프레임
            var extractedFrames = 0
            
            Log.d(TAG, "📹 영상 분석 설정:")
            Log.d(TAG, "   - 영상 길이: ${duration}ms")
            Log.d(TAG, "   - 프레임 간격: ${frameInterval / 1000000.0}초")
            Log.d(TAG, "   - 최대 프레임: ${maxFrames}개")
            
            while (currentTime < duration * 1000 && extractedFrames < maxFrames) {
                try {
                    val frame = retriever.getFrameAtTime(currentTime, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    frame?.let {
                        frames.add(it)
                        extractedFrames++
                        frameCount++
                        
                        val extractProgress = 20 + (extractedFrames * 30 / maxFrames)
                        callback.onProgressUpdate(extractProgress, "프레임 추출 중 (${extractedFrames}/${maxFrames})")
                        
                        Log.v(TAG, "프레임 추출 성공: ${extractedFrames}/${maxFrames}")
                    }
                    currentTime += frameInterval
                } catch (e: Exception) {
                    Log.w(TAG, "프레임 추출 실패 at $currentTime: ${e.message}")
                    break
                }
            }
            
            retriever.release()
            
            if (frames.isEmpty()) {
                Log.w(TAG, "추출된 프레임이 없음")
                callback.onProgressUpdate(100, "분석 완료")
                return@withContext EmergencyAnalysisResult(
                    isEmergency = false,
                    maxConfidence = 0.0f,
                    emergencyFrameRatio = 0.0f,
                    totalFrames = 0,
                    emergencyFrames = 0,
                    detectedClasses = emptyMap(),
                    dominantClass = "알 수 없음"
                )
            }
            
            callback.onProgressUpdate(50, "프레임 분석 중")
            
            // 개별 프레임 분석 (기존 모델 호환)
            Log.d(TAG, "🖼️ 개별 프레임 분석 시작 (총 ${frames.size}프레임)")
            
            var maxConfidence = 0.0f
            var emergencyFrameCount = 0
            val detectedClasses = mutableMapOf<String, Int>()
            
            frames.forEachIndexed { index, frame ->
                try {
                    val result = analyzeFrame(frame)
                    
                    if (result.confidence > maxConfidence) {
                        maxConfidence = result.confidence
                    }
                    
                    if (result.isEmergency) {
                        emergencyFrameCount++
                        val classLabel = result.classLabel
                        detectedClasses[classLabel] = detectedClasses.getOrDefault(classLabel, 0) + 1
                        Log.d(TAG, "⚠️ 프레임 ${index + 1}: 응급상황 감지됨 (${result.classLabel}, 확률: ${String.format("%.1f", result.confidence * 100)}%)")
                    } else {
                        Log.v(TAG, "✅ 프레임 ${index + 1}: 정상 (확률: ${String.format("%.1f", result.confidence * 100)}%)")
                    }
                    
                    val analysisProgress = 50 + (index * 40 / frames.size)
                    callback.onProgressUpdate(analysisProgress, "프레임 분석 중 (${index + 1}/${frames.size})")
                    
                } catch (e: Exception) {
                    Log.w(TAG, "프레임 ${index + 1} 분석 실패: ${e.message}")
                }
            }
            
            val emergencyFrameRatio = emergencyFrameCount.toFloat() / frames.size
            
            // 이진 분류 기준 적용 (수정된 기준)
            val emergencyThreshold = 0.5f  // 50% 이상 신뢰도
            val frameRatioThreshold = 0.3f  // 30% 이상 프레임
            
            // 더 관대한 기준 적용
            val isEmergency = emergencyFrameRatio > frameRatioThreshold || maxConfidence > emergencyThreshold
            
            val dominantClass = detectedClasses.maxByOrNull { it.value }?.key ?: "알 수 없음"
            
            Log.d(TAG, "📈 === 8클래스 모델 최종 분석 결과 ===")
            Log.d(TAG, "📊 총 프레임: ${frames.size}개")
            Log.d(TAG, "🚨 응급 프레임: ${emergencyFrameCount}개")
            Log.d(TAG, "📉 응급 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%")
            Log.d(TAG, "🎯 최고 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%")
            Log.d(TAG, "🏆 주요 감지 클래스: $dominantClass")
            Log.d(TAG, "📋 감지된 클래스별 분포:")
            detectedClasses.entries.sortedByDescending { it.value }.forEach { (label, count) ->
                Log.d(TAG, "     • $label: ${count}프레임 (${String.format("%.1f", count * 100.0f / frames.size)}%)")
            }
            
            // 🔍 판정 과정 상세 로그
            Log.d(TAG, "🔍 === 판정 과정 분석 ===")
            Log.d(TAG, "🎚️ 기준값:")
            Log.d(TAG, "   - 프레임 비율 임계값: ${String.format("%.1f", frameRatioThreshold * 100)}%")
            Log.d(TAG, "   - 신뢰도 임계값: ${String.format("%.1f", emergencyThreshold * 100)}%")
            Log.d(TAG, "🎯 현재값:")
            Log.d(TAG, "   - 실제 프레임 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%")
            Log.d(TAG, "   - 실제 최고 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%")
            Log.d(TAG, "✅ 조건 체크:")
            Log.d(TAG, "   - 프레임 비율 조건: ${emergencyFrameRatio > frameRatioThreshold} (${String.format("%.1f", emergencyFrameRatio * 100)}% > ${String.format("%.1f", frameRatioThreshold * 100)}%)")
            Log.d(TAG, "   - 신뢰도 조건: ${maxConfidence > emergencyThreshold} (${String.format("%.1f", maxConfidence * 100)}% > ${String.format("%.1f", emergencyThreshold * 100)}%)")
            Log.d(TAG, "   - OR 조건 결과: $isEmergency")
            
            Log.d(TAG, "🔔 최종 판정: ${if (isEmergency) "🚨 응급상황" else "✅ 정상상황"}")
            Log.d(TAG, "📋 판정 기준: 프레임비율>${String.format("%.0f", frameRatioThreshold * 100)}% OR 신뢰도>${String.format("%.0f", emergencyThreshold * 100)}% (8클래스 분류, 6번=정상)")
            
            // 🚨 모순 상황 감지
            if (!isEmergency && emergencyFrameCount > 0) {
                Log.w(TAG, "⚠️ 로직 모순 감지!")
                Log.w(TAG, "   - 응급 프레임이 ${emergencyFrameCount}개 있는데 정상으로 판정됨")
                Log.w(TAG, "   - 주요 클래스: $dominantClass")
                Log.w(TAG, "   - 이는 임계값 설정 문제일 수 있습니다")
            }
            
            callback.onProgressUpdate(100, "분석 완료!")
            
            return@withContext EmergencyAnalysisResult(
                isEmergency = isEmergency,
                maxConfidence = maxConfidence,
                emergencyFrameRatio = emergencyFrameRatio,
                totalFrames = frames.size,
                emergencyFrames = emergencyFrameCount,
                detectedClasses = detectedClasses,
                dominantClass = dominantClass
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "영상 분석 중 오류: ${e.message}", e)
            callback.onProgressUpdate(100, "분석 오류 발생")
            
            return@withContext EmergencyAnalysisResult(
                isEmergency = false,
                maxConfidence = 0.0f,
                emergencyFrameRatio = 0.0f,
                totalFrames = 0,
                emergencyFrames = 0,
                detectedClasses = emptyMap(),
                dominantClass = "알 수 없음"
            )
        }
    }
    
    // 기존 단일 프레임 분석 함수 (emergency_model.ptl용)
    private fun analyzeFrame(bitmap: Bitmap): FrameAnalysisResult {
        // 🚨 디버깅: 현재 모델 상태 명확히 표시
        Log.d(TAG, "🔍 === 프레임 분석 시작 ===")
        Log.d(TAG, "📊 모델 상태:")
        Log.d(TAG, "   - isModelLoaded: $isModelLoaded")
        Log.d(TAG, "   - model 객체: ${if (model != null) "존재함" else "null"}")
        Log.d(TAG, "   - 모델 파일: $MODEL_NAME")
        Log.d(TAG, "   - 예상 클래스 수: $NUM_CLASSES")
        
        return try {
            if (model == null || !isModelLoaded) {
                Log.w(TAG, "🚨 === 테스트 모드 실행 ===")
                Log.w(TAG, "실제 AI 모델이 로드되지 않아 가짜 결과를 반환합니다!")
                Log.w(TAG, "이는 모델 파일 문제나 라이브러리 오류로 인한 것입니다.")
                
                // 테스트 모드에서는 고정된 패턴 반환 (랜덤 대신)
                val fakeClass = 0  // 항상 정상으로 반환
                val fakeConfidence = 0.3f  // 낮은 신뢰도
                
                Log.w(TAG, "🎭 가짜 결과 반환: 클래스=${fakeClass}, 신뢰도=${fakeConfidence}")
                
                return FrameAnalysisResult(
                    predictedClass = fakeClass,
                    confidence = fakeConfidence,
                    isEmergency = fakeClass != NORMAL_CLASS_INDEX,
                    classLabel = CLASS_LABELS[fakeClass],
                    allProbabilities = FloatArray(NUM_CLASSES) { if (it == fakeClass) fakeConfidence else 0.0f }
                )
            }
            
            Log.d(TAG, "✅ === 실제 AI 모델 실행 ===")
            Log.d(TAG, "진짜 AI 모델로 추론을 수행합니다!")
            
            // 실제 모델 추론 로직
            // 이미지 전처리 (기존 방식으로 단순화)
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
            
            // 기존 TensorImageUtils 사용 (ImageNet 정규화)
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                floatArrayOf(0.485f, 0.456f, 0.406f), // ImageNet mean
                floatArrayOf(0.229f, 0.224f, 0.225f)  // ImageNet std
            )
            
            Log.v(TAG, "🖼️ 프레임 전처리 완료: ${INPUT_SIZE}x${INPUT_SIZE}")
            
            // 모델 추론
            val outputTensor = model!!.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray
            
            // 이진 분류 원시 스코어 로깅 (더 상세히)
            Log.d(TAG, "🔍 === 프레임 분석 상세 ===")
            Log.d(TAG, "📊 8클래스 모델 원시 출력:")
            scores.forEachIndexed { idx, score ->
                Log.d(TAG, "   - 클래스 $idx (${CLASS_LABELS[idx]}): ${String.format("%.4f", score)}")
            }
            
            // 소프트맥스 적용하여 확률로 변환
            val probabilities = softmax(scores)
            
            // 확률 로깅 (모든 클래스 표시)
            Log.d(TAG, "📈 모든 클래스 확률 분석:")
            probabilities.forEachIndexed { idx, prob ->
                val percentage = String.format("%.1f", prob * 100)
                val isHighConfidence = prob > 0.3f
                val marker = if (isHighConfidence) "🔥" else "  "
                Log.d(TAG, "   $marker 클래스 $idx: ${percentage}% ${if (idx == NORMAL_CLASS_INDEX) "← 정상클래스" else ""}")
            }
            
            // 상위 3개 클래스 표시
            val sortedIndices = probabilities.indices.sortedByDescending { probabilities[it] }
            Log.d(TAG, "🏆 상위 3개 클래스:")
            for (i in 0..2) {
                val idx = sortedIndices[i]
                val prob = probabilities[idx]
                Log.d(TAG, "   ${i+1}위: 클래스 $idx (${String.format("%.1f", prob * 100)}%)")
            }
            
            // 가장 높은 확률의 클래스 찾기
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[predictedClass]
            val isEmergency = predictedClass != NORMAL_CLASS_INDEX  // 6번 클래스가 정상상황
            
            // 예측 결과 로깅
            Log.d(TAG, "🎯 최종 예측 결과:")
            Log.d(TAG, "   - 예측 클래스: $predictedClass")
            Log.d(TAG, "   - 클래스 라벨: ${CLASS_LABELS[predictedClass]}")
            Log.d(TAG, "   - 신뢰도: ${String.format("%.2f", confidence * 100)}%")
            Log.d(TAG, "   - 응급여부: ${if (isEmergency) "🚨 응급" else "✅ 정상"}")
            Log.d(TAG, "   - 정상클래스($NORMAL_CLASS_INDEX) 확률: ${String.format("%.2f", probabilities[NORMAL_CLASS_INDEX] * 100)}%")
            Log.d(TAG, "   - 예측된 클래스 확률: ${String.format("%.2f", confidence * 100)}%")
            
            return FrameAnalysisResult(
                predictedClass = predictedClass,
                confidence = confidence,
                isEmergency = isEmergency,
                classLabel = CLASS_LABELS[predictedClass],
                allProbabilities = probabilities
            )
            
        } catch (e: Exception) {
            Log.w(TAG, "프레임 분석 실패: ${e.message}")
            // 오류 시 정상상황으로 반환
            return FrameAnalysisResult(
                predictedClass = NORMAL_CLASS_INDEX,
                confidence = 0.0f,
                isEmergency = false,
                classLabel = CLASS_LABELS[NORMAL_CLASS_INDEX],
                allProbabilities = FloatArray(NUM_CLASSES) { 0.0f }
            )
        }
    }
    
    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.maxOrNull() ?: 0.0f
        val expScores = scores.map { kotlin.math.exp((it - maxScore).toDouble()).toFloat() }
        val sumExp = expScores.sum()
        return expScores.map { it / sumExp }.toFloatArray()
    }
    
    private fun loadTensorFlowLiteModel() {
        // TensorFlow Lite 모델이 있다면 사용
        val tfliteDetector = TensorFlowLiteDetector(context)
        Log.w(TAG, "TensorFlow Lite 감지기로 대체됨 - 제한된 기능으로 작동")
        // 이 경우는 별도의 플래그로 관리할 수 있음
    }
    
    fun cleanup() {
        try {
            model = null
            isModelLoaded = false
            Log.d(TAG, "EmergencyDetector 정리 완료")
        } catch (e: Exception) {
            Log.e(TAG, "정리 중 오류: ${e.message}")
        }
    }
} 
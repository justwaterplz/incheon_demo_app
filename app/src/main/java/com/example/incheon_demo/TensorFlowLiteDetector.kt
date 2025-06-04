package com.example.incheon_demo

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer
import kotlin.random.Random

class TensorFlowLiteDetector(private val context: Context) {
    
    companion object {
        private const val TAG = "TensorFlowLiteDetector"
        private const val MODEL_NAME = "8cls.tflite"  // 8클래스 TFLite 모델
        private const val INPUT_SIZE = 200  // 원본 훈련 데이터 해상도에 맞춤 (224 -> 200)
        private const val NORMAL_CLASS_INDEX = 6  // 6번 클래스가 정상상황
        private const val NUM_CLASSES = 8  // 총 8개 클래스
        
        // 클래스 라벨 정의 (EmergencyDetector와 동일)
        private val CLASS_LABELS = arrayOf(
            "폭력행위", "낙상사고", "화재상황", "충돌사고",
            "절도행위", "이상행동", "정상상황", "응급상황"
        )
    }
    
    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private var imageProcessor: ImageProcessor? = null
    
    data class EmergencyAnalysisResult(
        val isEmergency: Boolean,
        val maxConfidence: Float,
        val emergencyFrameRatio: Float,
        val totalFrames: Int,
        val emergencyFrames: Int,
        val detectedClasses: Map<String, Int> = emptyMap(),  // 감지된 클래스별 프레임 수
        val dominantClass: String = "알 수 없음"  // 가장 많이 감지된 클래스
    )
    
    // 8클래스 분석 결과를 담는 데이터 클래스
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
            loadModel()
        } catch (e: Exception) {
            Log.e(TAG, "8클래스 TensorFlow Lite 모델 로딩 실패: ${e.message}", e)
        }
    }
    
    private fun loadModel() {
        try {
            Log.d(TAG, "8클래스 TensorFlow Lite 모델 로딩 시작: $MODEL_NAME")
            
            // assets에서 모델 파일 로드
            val modelBuffer: MappedByteBuffer = FileUtil.loadMappedFile(context, MODEL_NAME)
            
            // Interpreter 옵션 설정
            val options = Interpreter.Options()
            options.numThreads = 4  // 멀티스레드 사용
            
            // GPU 가속 시도 (실패하면 CPU 사용)
            try {
                // GPU delegate는 필요시 추가
                // val gpuDelegate = GpuDelegate()
                // options.addDelegate(gpuDelegate)
            } catch (e: Exception) {
                Log.w(TAG, "GPU delegate 사용 불가, CPU 사용: ${e.message}")
            }
            
            interpreter = Interpreter(modelBuffer, options)
            
            // 이미지 전처리기 설정
            imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .build()
            
            isModelLoaded = true
            Log.d(TAG, "✓ 8클래스 TensorFlow Lite 모델 로딩 성공")
            
        } catch (e: Exception) {
            Log.e(TAG, "8클래스 TensorFlow Lite 모델 로딩 실패: ${e.message}", e)
            isModelLoaded = false
            interpreter = null
        }
    }
    
    suspend fun detectFromVideoWithProgress(
        videoPath: String,
        callback: AnalysisProgressCallback
    ): EmergencyAnalysisResult = withContext(Dispatchers.IO) {
        
        if (!isModelLoaded || interpreter == null) {
            Log.w(TAG, "TensorFlow Lite 모델이 로드되지 않음 - 테스트 모드로 실행")
            callback.onProgressUpdate(100, "테스트 모드 완료")
            return@withContext EmergencyAnalysisResult(
                isEmergency = false,
                maxConfidence = 0.3f,
                emergencyFrameRatio = 0.0f,
                totalFrames = 10,
                emergencyFrames = 0,
                detectedClasses = emptyMap(),
                dominantClass = "알 수 없음"
            )
        }
        
        try {
            callback.onProgressUpdate(0, "영상 분석을 준비하고 있습니다")
            
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoPath)
            
            // 영상 길이 확인
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val duration = durationStr?.toLongOrNull() ?: 10000L // 기본값 10초
            
            callback.onProgressUpdate(10, "영상 정보 분석 완료")
            
            // 2초마다 프레임 추출
            val frameInterval = 2000000L // 2초 (마이크로초 단위)
            val frames = mutableListOf<Bitmap>()
            
            var currentTime = 0L
            var frameCount = 0
            val totalExpectedFrames = (duration / 2000).toInt().coerceAtLeast(1)
            
            callback.onProgressUpdate(20, "프레임 추출 중")
            
            while (currentTime < duration * 1000) { // duration을 마이크로초로 변환
                try {
                    val frame = retriever.getFrameAtTime(currentTime, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    frame?.let {
                        frames.add(it)
                        frameCount++
                        
                        val extractProgress = 20 + (frameCount * 30 / totalExpectedFrames)
                        callback.onProgressUpdate(extractProgress, "프레임 추출 중 ($frameCount/$totalExpectedFrames)")
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
            
            // 각 프레임에 대해 응급상황 분석
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
                        detectedClasses[result.classLabel] = detectedClasses.getOrDefault(result.classLabel, 0) + 1
                    }
                    
                    val analysisProgress = 50 + (index * 40 / frames.size)
                    callback.onProgressUpdate(analysisProgress, "프레임 분석 중 (${index + 1}/${frames.size})")
                    
                } catch (e: Exception) {
                    Log.w(TAG, "프레임 분석 실패: ${e.message}")
                }
            }
            
            val emergencyFrameRatio = emergencyFrameCount.toFloat() / frames.size
            
            // 더 엄격한 기준 적용
            val isEmergency = emergencyFrameRatio > 0.5f && maxConfidence > 0.8f
            
            val dominantClass = if (detectedClasses.isNotEmpty()) {
                detectedClasses.maxByOrNull { it.value }?.key ?: "알 수 없음"
            } else {
                "알 수 없음"
            }
            
            Log.d(TAG, "📈 === TensorFlow Lite 최종 분석 결과 ===")
            Log.d(TAG, "📊 총 프레임: ${frames.size}개")
            Log.d(TAG, "🚨 응급 프레임: ${emergencyFrameCount}개")
            Log.d(TAG, "📉 응급 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%")
            Log.d(TAG, "🎯 최고 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%")
            Log.d(TAG, "🔔 최종 판정: ${if (isEmergency) "🚨 응급상황" else "✅ 정상상황"}")
            Log.d(TAG, "📋 기준: 응급비율 > 50% AND 최고신뢰도 > 80%")
            
            callback.onProgressUpdate(100, "분석 완료")
            
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
            Log.e(TAG, "TensorFlow Lite 영상 분석 중 오류: ${e.message}", e)
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
    
    private fun analyzeFrame(bitmap: Bitmap): FrameAnalysisResult {
        // 🚨 디버깅: TensorFlow Lite 모델 상태 명확히 표시
        Log.d(TAG, "🔍 === TensorFlow Lite 프레임 분석 시작 ===")
        Log.d(TAG, "📊 TensorFlow Lite 모델 상태:")
        Log.d(TAG, "   - isModelLoaded: $isModelLoaded")
        Log.d(TAG, "   - interpreter: ${if (interpreter != null) "존재함" else "null"}")
        Log.d(TAG, "   - imageProcessor: ${if (imageProcessor != null) "존재함" else "null"}")
        Log.d(TAG, "   - 모델 파일: $MODEL_NAME")
        Log.d(TAG, "   - 예상 클래스 수: $NUM_CLASSES")
        
        return try {
            val interpreter = this.interpreter
            if (interpreter == null || imageProcessor == null) {
                Log.w(TAG, "🚨 === TensorFlow Lite 테스트 모드 실행 ===")
                Log.w(TAG, "실제 TensorFlow Lite 모델이 로드되지 않아 가짜 결과를 반환합니다!")
                Log.w(TAG, "이는 .tflite 모델 파일이 없거나 라이브러리 오류로 인한 것입니다.")
                
                // 테스트 모드: 랜덤 값 반환
                val randomClass = Random.nextInt(NUM_CLASSES)
                val randomConfidence = Random.nextFloat() * 0.3f + 0.1f // 0.1 ~ 0.4 사이의 값
                val isEmergency = randomClass != NORMAL_CLASS_INDEX
                val classLabel = CLASS_LABELS[randomClass]
                val allProbabilities = FloatArray(NUM_CLASSES) { if (it == randomClass) randomConfidence else 0.0f }
                
                Log.w(TAG, "🎭 가짜 랜덤 결과: 클래스=${randomClass}(${classLabel}), 신뢰도=${randomConfidence}, 응급=${isEmergency}")
                
                return FrameAnalysisResult(
                    predictedClass = randomClass,
                    confidence = randomConfidence,
                    isEmergency = isEmergency,
                    classLabel = classLabel,
                    allProbabilities = allProbabilities
                )
            }
            
            Log.d(TAG, "✅ === 실제 TensorFlow Lite 모델 실행 ===")
            Log.d(TAG, "진짜 TensorFlow Lite 모델로 추론을 수행합니다!")
            
            // 이미지 전처리
            val tensorImage = TensorImage.fromBitmap(bitmap)
            val processedImage = imageProcessor!!.process(tensorImage)
            
            // 입력 버퍼 생성
            val inputBuffer = processedImage.buffer
            
            // 출력 버퍼 생성
            val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, NUM_CLASSES), org.tensorflow.lite.DataType.FLOAT32)
            
            // 추론 실행
            interpreter.run(inputBuffer, outputBuffer.buffer)
            
            // 결과 가져오기
            val scores = outputBuffer.floatArray
            
            // 소프트맥스 적용하여 확률로 변환
            val probabilities = softmax(scores)
            
            // 가장 높은 확률의 클래스 찾기
            val predictedClass = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val confidence = probabilities[predictedClass]
            val isEmergency = predictedClass != NORMAL_CLASS_INDEX
            val classLabel = CLASS_LABELS[predictedClass]
            val allProbabilities = probabilities
            
            Log.v(TAG, "TensorFlow Lite 프레임 분석 결과: 응급상황 확률 = $confidence")
            
            return FrameAnalysisResult(
                predictedClass = predictedClass,
                confidence = confidence,
                isEmergency = isEmergency,
                classLabel = classLabel,
                allProbabilities = allProbabilities
            )
            
        } catch (e: Exception) {
            Log.w(TAG, "TensorFlow Lite 프레임 분석 실패: ${e.message}")
            return FrameAnalysisResult(
                predictedClass = -1,
                confidence = 0.0f,
                isEmergency = false,
                classLabel = "알 수 없음",
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
    
    fun cleanup() {
        try {
            interpreter?.close()
            interpreter = null
            isModelLoaded = false
            Log.d(TAG, "TensorFlow Lite Detector 정리 완료")
        } catch (e: Exception) {
            Log.e(TAG, "정리 중 오류: ${e.message}")
        }
    }
} 
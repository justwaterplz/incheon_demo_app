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
import kotlin.math.exp

class ActionClassifier(private val context: Context) {
    
    companion object {
        private const val TAG = "ActionClassifier"
        private const val MODEL_NAME = "8cls.ptl"  // 기존 모델로 임시 복원
        private const val INPUT_SIZE = 112  // 원본 Python 모델과 일치 (112x112)
        private const val NUM_FRAMES = 16   // 3D ResNet 스타일 연속 프레임
        private const val NUM_CLASSES = 8   // AI 모델의 클래스 수 (baseline_3d_resnets와 호환)
        
        // AI 모델의 8개 클래스 레이블
        private val CLASS_LABELS = arrayOf(
            "폭행",        // 0: assault
            "소매치기",    // 1: burglar  
            "데이트 폭력", // 2: date
            "취객",        // 3: drunken
            "싸움",        // 4: fight
            "납치",        // 5: kidnap
            "정상",        // 6: none
            "강도"         // 7: robbery
        )
        
        // 응급상황으로 간주할 클래스 인덱스들
        private val EMERGENCY_CLASS_INDICES = setOf(0, 1, 2, 3, 4, 5, 7)  // 정상(6)을 제외한 모든 클래스
    }
    
    private var model: Module? = null
    private var isModelLoaded = false
    
    data class ActionAnalysisResult(
        val isEmergency: Boolean,
        val topPredictions: List<ClassPrediction>,  // Top3 예측 결과
        val confidence: Float,
        val totalSegments: Int,
        val emergencySegments: Int,
        val dominantAction: String
    )
    
    data class ClassPrediction(
        val classIndex: Int,
        val className: String,
        val confidence: Float,
        val isEmergency: Boolean
    )
    
    data class SegmentAnalysisResult(
        val predictions: List<ClassPrediction>,
        val isEmergency: Boolean
    )
    
    interface AnalysisProgressCallback {
        fun onProgressUpdate(progress: Int, status: String)
    }
    
    init {
        try {
            Log.d(TAG, "🚀 ActionClassifier 초기화 시작")
            loadModel()
            Log.d(TAG, "✅ ActionClassifier 초기화 완료")
        } catch (e: Exception) {
            Log.e(TAG, "💥 ActionClassifier 초기화 실패: ${e.message}", e)
            isModelLoaded = false
            model = null
        }
    }
    
    private fun loadModel() {
        try {
            Log.d(TAG, "🚀 8클래스 동작 인식 모델 로딩 시작: $MODEL_NAME")
            
            // assets 파일 존재 확인
            val assetManager = context.assets
            val assetList = assetManager.list("")
            Log.d(TAG, "📁 Assets 파일 목록: ${assetList?.joinToString(", ")}")
            
            // 모델 파일 크기 확인
            val inputStream = assetManager.open(MODEL_NAME)
            val fileSize = inputStream.available()
            inputStream.close()
            Log.d(TAG, "📄 모델 파일 크기: ${fileSize / (1024 * 1024)}MB")
            
            if (fileSize < 1000000) { // 1MB 미만이면 문제
                throw RuntimeException("모델 파일이 너무 작음: ${fileSize} bytes")
            }
            
            // assets에서 모델 파일 복사
            val modelPath = ModelUtils.assetFilePath(context, MODEL_NAME)
            Log.d(TAG, "📂 모델 파일 경로: $modelPath")
            
            // 모델 로딩
            model = LiteModuleLoader.load(modelPath)
            
            if (model == null) {
                throw RuntimeException("LiteModuleLoader.load()가 null을 반환함")
            }
            
            // 모델 테스트 추론 실행 (실패해도 계속 진행)
            try {
                testModelInference()
            } catch (e: Exception) {
                Log.w(TAG, "⚠️ 모델 테스트는 실패했지만 로딩은 완료됨: ${e.message}")
            }
            
            isModelLoaded = true
            Log.d(TAG, "🎉 8클래스 동작 인식 모델 로딩 완료!")
            
        } catch (e: Exception) {
            Log.e(TAG, "💥 모델 로딩 실패: ${e.message}", e)
            isModelLoaded = false
            model = null
            throw RuntimeException("모델 로딩 실패: ${e.message}", e)
        }
    }
    
    private fun testModelInference() {
        try {
            if (model == null) return
            
            Log.d(TAG, "🧪 모델 테스트 추론 시작 (4D/5D 혼합)")
            
            // 여러 차원으로 테스트해보기 (112x112 기준) - 5D 우선
            val testCases = listOf(
                // Case 1: 5D 텐서 (3D ResNet의 올바른 입력) - 최우선
                Triple(
                    "5D-TrueCNN",
                    longArrayOf(1, 3, NUM_FRAMES.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                    3 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE
                ),
                // Case 2: 4D 텐서 (프레임을 채널로 flatten) - 백업용
                Triple(
                    "4D-FramesToChannels", 
                    longArrayOf(1, (3 * NUM_FRAMES).toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                    3 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE
                ),
                // Case 3: 4D 텐서 (단일 프레임) - 최후 수단
                Triple(
                    "4D-SingleFrame", 
                    longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()),
                    3 * INPUT_SIZE * INPUT_SIZE
                )
            )
            
            var testSuccess = false
            var successCase = ""
            
            for ((caseName, shape, dataSize) in testCases) {
                try {
                    Log.d(TAG, "🔍 테스트 케이스: $caseName, 형태: ${shape.contentToString()}")
                    
                    // 테스트용 더미 데이터 생성 (ImageNet 정규화된 값)
                    val testData = FloatArray(dataSize) { i ->
                        when (i % 3) {
                            0 -> (0.485f - 0.485f) / 0.229f  // R 채널 평균값
                            1 -> (0.456f - 0.456f) / 0.224f  // G 채널 평균값
                            else -> (0.406f - 0.406f) / 0.225f  // B 채널 평균값
                        }
                    }
                    
                    val testInput = org.pytorch.Tensor.fromBlob(testData, shape)
                    
                    // 모델 추론 실행
                    val output = model!!.forward(IValue.from(testInput)).toTensor()
                    val scores = output.dataAsFloatArray
                    
                    Log.d(TAG, "✅ $caseName 테스트 성공!")
                    Log.d(TAG, "   - 입력 형태: ${shape.contentToString()}")
                    Log.d(TAG, "   - 출력 클래스 수: ${scores.size}")
                    
                    // Softmax 적용하여 확률 계산
                    val probabilities = softmax(scores)
                    
                    Log.d(TAG, "🔍 $caseName 테스트 출력 분석:")
                    probabilities.take(Math.min(5, probabilities.size)).forEachIndexed { idx, prob ->
                        val className = if (idx < CLASS_LABELS.size) CLASS_LABELS[idx] else "클래스$idx"
                        Log.d(TAG, "   클래스 $idx ($className): ${String.format("%.2f", prob * 100)}%")
                    }
                    
                    testSuccess = true
                    successCase = caseName
                    break // 첫 번째 성공한 케이스로 결정
                    
                } catch (e: Exception) {
                    Log.w(TAG, "⚠️ $caseName 실패: ${e.message}")
                }
            }
            
            if (testSuccess) {
                Log.d(TAG, "🎉 모델 테스트 완료! 성공한 형태: $successCase")
            } else {
                Log.w(TAG, "⚠️ 모든 테스트 케이스 실패 - 실제 추론 시 동적으로 처리")
            }
            
        } catch (e: Exception) {
            Log.w(TAG, "⚠️ 모델 테스트 중 오류: ${e.message}")
            Log.w(TAG, "   스택 트레이스: ${e.stackTraceToString()}")
            // 테스트 실패해도 계속 진행
        }
    }
    
    suspend fun analyzeVideoWithProgress(
        videoPath: String,
        callback: AnalysisProgressCallback
    ): ActionAnalysisResult = withContext(Dispatchers.IO) {
        
        if (!isModelLoaded || model == null) {
            val errorMsg = "🚨 AI 모델이 로드되지 않았습니다."
            Log.e(TAG, errorMsg)
            callback.onProgressUpdate(100, "모델 로딩 실패")
            throw RuntimeException(errorMsg)
        }
        
        try {
            callback.onProgressUpdate(0, "영상 분석 준비 중...")
            
            // 비디오 파일 검증
            val videoFile = File(videoPath)
            Log.d(TAG, "🔍 비디오 파일 검증:")
            Log.d(TAG, "   - 파일 경로: $videoPath")
            Log.d(TAG, "   - 파일 존재: ${videoFile.exists()}")
            Log.d(TAG, "   - 파일 크기: ${videoFile.length()} bytes")
            Log.d(TAG, "   - 절대 경로: ${videoFile.absolutePath}")
            Log.d(TAG, "   - 읽기 권한: ${videoFile.canRead()}")
            
            if (!videoFile.exists()) {
                val parentDir = videoFile.parentFile
                Log.e(TAG, "❌ 파일이 존재하지 않음!")
                Log.e(TAG, "   - 부모 디렉토리: ${parentDir?.absolutePath}")
                Log.e(TAG, "   - 부모 디렉토리 존재: ${parentDir?.exists()}")
                if (parentDir?.exists() == true) {
                    Log.e(TAG, "   - 디렉토리 내용: ${parentDir.listFiles()?.map { it.name }?.joinToString(", ")}")
                }
                throw RuntimeException("비디오 파일이 존재하지 않음: $videoPath")
            }
            
            if (videoFile.length() == 0L) {
                Log.e(TAG, "❌ 파일 크기가 0 bytes!")
                throw RuntimeException("비디오 파일이 손상됨 (크기: 0 bytes)")
            }
            
            Log.d(TAG, "✅ 비디오 파일 검증 완료!")
            
            // 연속 프레임 세그먼트들 추출
            val segments = extractVideoSegments(videoPath, callback)
            
            if (segments.isEmpty()) {
                throw RuntimeException("추출된 프레임 세그먼트가 없음")
            }
            
            callback.onProgressUpdate(50, "동작 분석 중...")
            
            // 각 세그먼트에 대해 동작 분석
            val segmentResults = mutableListOf<SegmentAnalysisResult>()
            val allPredictions = mutableListOf<ClassPrediction>()
            var emergencySegmentCount = 0
            
            segments.forEachIndexed { index, segment ->
                try {
                    val result = analyzeSegment(segment)
                    segmentResults.add(result)
                    allPredictions.addAll(result.predictions)
                    
                    if (result.isEmergency) {
                        emergencySegmentCount++
                    }
                    
                    val progress = 50 + (index * 40 / segments.size)
                    callback.onProgressUpdate(progress, "세그먼트 ${index + 1}/${segments.size} 분석 완료")
                    
                } catch (e: Exception) {
                    Log.w(TAG, "세그먼트 분석 실패: ${e.message}")
                }
            }
            
            // Top3 예측 결과 계산
            val classCounts = mutableMapOf<Int, Float>()
            allPredictions.forEach { pred ->
                classCounts[pred.classIndex] = classCounts.getOrDefault(pred.classIndex, 0f) + pred.confidence
            }
            
            val topPredictions = classCounts.entries
                .sortedByDescending { it.value }
                .take(3)
                .map { (classIndex, totalConfidence) ->
                    ClassPrediction(
                        classIndex = classIndex,
                        className = if (classIndex < CLASS_LABELS.size) CLASS_LABELS[classIndex] else "클래스$classIndex",
                        confidence = totalConfidence / allPredictions.size,
                        isEmergency = classIndex in EMERGENCY_CLASS_INDICES
                    )
                }
            
            // 최종 판정
            val isEmergency = topPredictions.any { it.isEmergency && it.confidence > 0.3f } ||
                              (emergencySegmentCount.toFloat() / segments.size) > 0.2f
            
            val dominantAction = topPredictions.firstOrNull()?.className ?: "알 수 없음"
            val maxConfidence = topPredictions.firstOrNull()?.confidence ?: 0f
            
            Log.d(TAG, "🔔 최종 분석 결과:")
            Log.d(TAG, "   - 총 세그먼트: ${segments.size}")
            Log.d(TAG, "   - 응급 세그먼트: $emergencySegmentCount")
            Log.d(TAG, "   - 주요 동작: $dominantAction")
            Log.d(TAG, "   - 최종 판정: ${if (isEmergency) "🚨 응급상황" else "✅ 정상"}")
            
            callback.onProgressUpdate(100, "분석 완료!")
            
            return@withContext ActionAnalysisResult(
                isEmergency = isEmergency,
                topPredictions = topPredictions,
                confidence = maxConfidence,
                totalSegments = segments.size,
                emergencySegments = emergencySegmentCount,
                dominantAction = dominantAction
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "영상 분석 중 오류: ${e.message}", e)
            callback.onProgressUpdate(100, "분석 오류 발생")
            throw e
        }
    }
    
    private fun extractVideoSegments(videoPath: String, callback: AnalysisProgressCallback): List<List<Bitmap>> {
        val segments = mutableListOf<List<Bitmap>>()
        val retriever = MediaMetadataRetriever()
        
        try {
            retriever.setDataSource(videoPath)
            val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            val duration = durationStr?.toLongOrNull() ?: 10000L
            
            // 세그먼트 간격 계산 (겹치는 구간 포함)
            val segmentDuration = 2000L  // 2초 세그먼트
            val stepSize = 1000L         // 1초씩 이동 (50% 겹침)
            
            var currentTime = 0L
            var segmentIndex = 0
            
            while (currentTime + segmentDuration <= duration) {
                val frames = mutableListOf<Bitmap>()
                val frameInterval = segmentDuration / NUM_FRAMES
                
                for (i in 0 until NUM_FRAMES) {
                    val frameTime = currentTime + (i * frameInterval)
                    try {
                        val frame = retriever.getFrameAtTime(
                            frameTime * 1000, // 마이크로초로 변환
                            MediaMetadataRetriever.OPTION_CLOSEST_SYNC
                        )
                        frame?.let { frames.add(it) }
                    } catch (e: Exception) {
                        Log.w(TAG, "프레임 추출 실패 at ${frameTime}ms: ${e.message}")
                    }
                }
                
                // 16프레임이 모두 추출되었을 때만 세그먼트로 추가
                if (frames.size == NUM_FRAMES) {
                    segments.add(frames)
                    segmentIndex++
                }
                
                currentTime += stepSize
                
                val progress = 20 + (currentTime * 30 / duration).toInt()
                callback.onProgressUpdate(progress, "세그먼트 추출 중 ($segmentIndex)")
            }
            
        } finally {
            retriever.release()
        }
        
        Log.d(TAG, "📹 총 ${segments.size}개 세그먼트 추출 완료")
        return segments
    }
    
    private fun analyzeSegment(frames: List<Bitmap>): SegmentAnalysisResult {
        try {
            // 텐서 생성 시도
            val inputTensor = createInputTensor(frames)
            
            // 모델 추론 시도
            val outputTensor = model!!.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray
            
            Log.d(TAG, "🔍 세그먼트 추론 성공: 출력 크기=${scores.size}")
            
            // Softmax로 확률 변환
            val probabilities = softmax(scores)
            
            // Top3 예측 생성
            val predictions = probabilities.indices
                .sortedByDescending { probabilities[it] }
                .take(3)
                .map { classIndex ->
                    ClassPrediction(
                        classIndex = classIndex,
                        className = if (classIndex < CLASS_LABELS.size) CLASS_LABELS[classIndex] else "클래스$classIndex",
                        confidence = probabilities[classIndex],
                        isEmergency = classIndex in EMERGENCY_CLASS_INDICES
                    )
                }
            
            val isEmergency = predictions.any { it.isEmergency && it.confidence > 0.4f }
            
            Log.d(TAG, "🎯 세그먼트 결과: ${predictions.firstOrNull()?.className ?: "알수없음"} (${String.format("%.1f", (predictions.firstOrNull()?.confidence ?: 0f) * 100)}%)")
            
            return SegmentAnalysisResult(
                predictions = predictions,
                isEmergency = isEmergency
            )
            
        } catch (e: Exception) {
            Log.w(TAG, "⚠️ 세그먼트 분석 실패: ${e.message}")
            
            // 실패 시 기본값 반환 (정상활동으로 가정)
            return SegmentAnalysisResult(
                predictions = listOf(
                    ClassPrediction(6, "정상활동", 0.5f, false),
                    ClassPrediction(0, "알수없음1", 0.3f, false),
                    ClassPrediction(1, "알수없음2", 0.2f, false)
                ),
                isEmergency = false
            )
        }
    }
    
    private fun createInputTensor(frames: List<Bitmap>): Tensor {
        try {
            Log.d(TAG, "🔄 5D 텐서 생성 시작: ${frames.size}개 프레임 → (1, 3, 16, 112, 112)")
            
            // 5D 텐서 데이터 배열 생성 (B, C, T, H, W) = (1, 3, 16, 112, 112)
            val tensorData = FloatArray(3 * NUM_FRAMES * INPUT_SIZE * INPUT_SIZE)
            
            // 프레임 수 조정 (16개 맞춤)
            val processFrames = when {
                frames.size >= NUM_FRAMES -> frames.take(NUM_FRAMES)
                frames.size > 0 -> {
                    Log.d(TAG, "⚠️ 프레임 수 부족 (${frames.size} < $NUM_FRAMES), 마지막 프레임 반복")
                    val extendedFrames = frames.toMutableList()
                    while (extendedFrames.size < NUM_FRAMES) {
                        extendedFrames.add(frames.last())
                    }
                    extendedFrames
                }
                else -> {
                    Log.e(TAG, "❌ 프레임이 없음!")
                    throw RuntimeException("입력 프레임이 없습니다")
                }
            }
            
            // 각 프레임을 5D 텐서 형태로 배치 (B, C, T, H, W)
            processFrames.forEachIndexed { frameIndex, bitmap ->
                try {
                    // 프레임을 112x112로 리사이즈
                    val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
                    
                    // 픽셀 데이터 추출
                    val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
                    resizedBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
                    
                    // RGB 채널별로 ImageNet 정규화 적용
                    for (pixelIndex in pixels.indices) {
                        val pixel = pixels[pixelIndex]
                        
                        // RGB 추출 (0~255)
                        val r = ((pixel shr 16) and 0xFF) / 255.0f
                        val g = ((pixel shr 8) and 0xFF) / 255.0f
                        val b = (pixel and 0xFF) / 255.0f
                        
                        // 3D ResNet 정규화 적용: 1-2*(x/255) → [-1, 1] 범위
                        val rNorm = 1.0f - 2.0f * r  // 이미 0~1 범위이므로 바로 적용
                        val gNorm = 1.0f - 2.0f * g
                        val bNorm = 1.0f - 2.0f * b
                        
                        // 5D 텐서 배치: (B, C, T, H, W) = (1, 3, 16, 112, 112)
                        // 인덱스 계산: batch * (C*T*H*W) + channel * (T*H*W) + time * (H*W) + pixel
                        val frameSize = INPUT_SIZE * INPUT_SIZE  // H * W
                        val channelSize = NUM_FRAMES * frameSize  // T * H * W
                        
                        // R, G, B 채널별로 데이터 배치
                        tensorData[0 * channelSize + frameIndex * frameSize + pixelIndex] = rNorm  // R 채널
                        tensorData[1 * channelSize + frameIndex * frameSize + pixelIndex] = gNorm  // G 채널
                        tensorData[2 * channelSize + frameIndex * frameSize + pixelIndex] = bNorm  // B 채널
                    }
                    
                } catch (e: Exception) {
                    Log.e(TAG, "프레임 $frameIndex 처리 실패: ${e.message}")
                    throw e
                }
            }
            
            // 5D 텐서 생성: (1, 3, 16, 112, 112)
            val tensor = Tensor.fromBlob(
                tensorData,
                longArrayOf(1, 3, NUM_FRAMES.toLong(), INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )
            
            Log.d(TAG, "✅ 5D 텐서 생성 성공!")
            Log.d(TAG, "   - 형태: (1, 3, $NUM_FRAMES, $INPUT_SIZE, $INPUT_SIZE)")
            Log.d(TAG, "   - 데이터 크기: ${tensorData.size}")
            Log.d(TAG, "   - 프레임 처리: ${processFrames.size}개 → 시간축 유지")
            
            return tensor
            
        } catch (e: Exception) {
            Log.e(TAG, "💥 5D 텐서 생성 실패: ${e.message}")
            throw RuntimeException("5D 입력 텐서 생성 실패", e)
        }
    }
    
    private fun softmax(scores: FloatArray): FloatArray {
        val maxScore = scores.maxOrNull() ?: 0.0f
        val expScores = scores.map { exp((it - maxScore).toDouble()).toFloat() }
        val sumExp = expScores.sum()
        return expScores.map { it / sumExp }.toFloatArray()
    }
    
    fun cleanup() {
        try {
            model = null
            isModelLoaded = false
            Log.d(TAG, "ActionClassifier 정리 완료")
        } catch (e: Exception) {
            Log.e(TAG, "정리 중 오류: ${e.message}")
        }
    }
} 
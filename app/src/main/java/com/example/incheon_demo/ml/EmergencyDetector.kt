package com.example.incheon_demo.ml

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.util.Log
import com.example.incheon_demo.utils.ModelUtils
import org.pytorch.IValue
import org.pytorch.Module

class EmergencyDetector(context: Context) {
    private var model: Module? = null
    
    // 진행 상황을 알리기 위한 콜백 인터페이스
    interface AnalysisProgressCallback {
        fun onProgressUpdate(progress: Int, status: String)
        fun onAnalysisComplete(result: EmergencyAnalysisResult)
        fun onError(error: String)
    }
    
    init {
        try {
            // model_lite.ptl 파일을 로드 (아직 변환하지 않았다면 실패할 것임)
            model = ModelUtils.loadModelFromAssets(context, "model_lite.ptl")
            Log.d(TAG, "모델 로드 성공")
        } catch (e: Exception) {
            Log.e(TAG, "모델 로드 실패: ${e.message}", e)
            model = null
        }
    }
    
    // 비디오 프레임에서 응급 상황 감지 (진행 상황 콜백 포함)
    fun detectFromVideoWithProgress(videoPath: String, callback: AnalysisProgressCallback) {
        try {
            callback.onProgressUpdate(0, "영상 분석을 준비하고 있습니다...")
            
            if (model == null) {
                Log.w(TAG, "모델이 로드되지 않음. 더미 결과 반환")
                callback.onProgressUpdate(50, "모델 파일이 없어 테스트 모드로 진행합니다...")
                
                // 테스트용 지연
                Thread.sleep(2000)
                callback.onProgressUpdate(100, "분석 완료")
                
                val result = EmergencyAnalysisResult(
                    isEmergency = false,
                    emergencyFrameRatio = 0f,
                    maxConfidence = 0f
                )
                callback.onAnalysisComplete(result)
                return
            }
            
            val retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoPath)
            
            var emergencyFrameCount = 0
            var totalFrames = 0
            var maxEmergencyConfidence = 0f
            
            try {
                callback.onProgressUpdate(10, "영상 정보를 분석하고 있습니다...")
                
                // 비디오의 프레임 수 계산
                val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
                val frameInterval = 2000L // 2초마다 프레임 추출 (성능 고려)
                val expectedFrames = (duration / frameInterval).toInt() + 1
                
                Log.d(TAG, "비디오 분석 시작: duration=${duration}ms, expectedFrames=$expectedFrames")
                callback.onProgressUpdate(20, "총 ${expectedFrames}개 프레임을 분석합니다...")
                
                for (timeMs in 0L..duration step frameInterval) {
                    try {
                        // 프레임 추출
                        val bitmap = retriever.getFrameAtTime(timeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                        bitmap?.let {
                            // 프레임에서 응급 상황 감지
                            val result = detectFromFrame(it)
                            if (result.isEmergency) {
                                emergencyFrameCount++
                                maxEmergencyConfidence = maxOf(maxEmergencyConfidence, result.confidence)
                            }
                            totalFrames++
                            it.recycle()
                            
                            // 진행 상황 업데이트
                            val progress = 20 + ((totalFrames.toFloat() / expectedFrames.toFloat()) * 70).toInt()
                            callback.onProgressUpdate(progress, "${totalFrames}/${expectedFrames} 프레임 분석 중...")
                            
                            Log.d(TAG, "프레임 $totalFrames 분석 완료: emergency=${result.isEmergency}, confidence=${result.confidence}")
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "프레임 추출/분석 오류: ${e.message}")
                    }
                }
                
                callback.onProgressUpdate(95, "분석 결과를 처리하고 있습니다...")
                
            } finally {
                retriever.release()
            }
            
            // 전체 영상에 대한 분석 결과 생성
            val emergencyRatio = if (totalFrames > 0) emergencyFrameCount.toFloat() / totalFrames.toFloat() else 0f
            val isEmergency = emergencyRatio > EMERGENCY_RATIO_THRESHOLD && maxEmergencyConfidence > CONFIDENCE_THRESHOLD
            
            Log.d(TAG, "비디오 분석 완료: frames=$totalFrames, emergencyFrames=$emergencyFrameCount, ratio=$emergencyRatio, maxConf=$maxEmergencyConfidence, isEmergency=$isEmergency")
            
            callback.onProgressUpdate(100, "분석 완료!")
            
            val result = EmergencyAnalysisResult(
                isEmergency = isEmergency,
                emergencyFrameRatio = emergencyRatio,
                maxConfidence = maxEmergencyConfidence
            )
            
            callback.onAnalysisComplete(result)
            
        } catch (e: Exception) {
            Log.e(TAG, "영상 분석 중 오류 발생: ${e.message}", e)
            callback.onError("영상 분석 중 오류가 발생했습니다: ${e.message}")
        }
    }
    
    // 기존 메서드는 유지 (호환성을 위해)
    fun detectFromVideo(videoPath: String): EmergencyAnalysisResult {
        if (model == null) {
            Log.w(TAG, "모델이 로드되지 않음. 더미 결과 반환")
            return EmergencyAnalysisResult(
                isEmergency = false,
                emergencyFrameRatio = 0f,
                maxConfidence = 0f
            )
        }
        
        val retriever = MediaMetadataRetriever()
        retriever.setDataSource(videoPath)
        
        var emergencyFrameCount = 0
        var totalFrames = 0
        var maxEmergencyConfidence = 0f
        
        try {
            // 비디오의 프레임 수 계산
            val duration = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: 0L
            val frameInterval = 2000L // 2초마다 프레임 추출 (성능 고려)
            
            Log.d(TAG, "비디오 분석 시작: duration=${duration}ms")
            
            for (timeMs in 0L..duration step frameInterval) {
                try {
                    // 프레임 추출
                    val bitmap = retriever.getFrameAtTime(timeMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    bitmap?.let {
                        // 프레임에서 응급 상황 감지
                        val result = detectFromFrame(it)
                        if (result.isEmergency) {
                            emergencyFrameCount++
                            maxEmergencyConfidence = maxOf(maxEmergencyConfidence, result.confidence)
                        }
                        totalFrames++
                        it.recycle()
                        
                        Log.d(TAG, "프레임 $totalFrames 분석 완료: emergency=${result.isEmergency}, confidence=${result.confidence}")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "프레임 추출/분석 오류: ${e.message}")
                }
            }
        } finally {
            retriever.release()
        }
        
        // 전체 영상에 대한 분석 결과 생성
        val emergencyRatio = if (totalFrames > 0) emergencyFrameCount.toFloat() / totalFrames.toFloat() else 0f
        val isEmergency = emergencyRatio > EMERGENCY_RATIO_THRESHOLD && maxEmergencyConfidence > CONFIDENCE_THRESHOLD
        
        Log.d(TAG, "비디오 분석 완료: frames=$totalFrames, emergencyFrames=$emergencyFrameCount, ratio=$emergencyRatio, maxConf=$maxEmergencyConfidence, isEmergency=$isEmergency")
        
        return EmergencyAnalysisResult(
            isEmergency = isEmergency,
            emergencyFrameRatio = emergencyRatio,
            maxConfidence = maxEmergencyConfidence
        )
    }
    
    // 단일 프레임에서 응급 상황 감지
    private fun detectFromFrame(bitmap: Bitmap): DetectionResult {
        if (model == null) {
            // 모델이 없을 때는 더미 결과 반환
            return DetectionResult(
                isEmergency = false,
                confidence = 0f
            )
        }
        
        try {
            // 비트맵을 224x224로 리사이즈
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            
            // 입력 텐서 준비
            val inputTensor = ModelUtils.bitmapToFloat32Tensor(resizedBitmap)
            
            // 모델 추론 실행
            val outputTensor = model!!.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray
            
            // 이진 분류 결과 해석 (응급/비응급)
            // scores[0]: 비응급 확률, scores[1]: 응급 확률
            val emergencyProbability = if (scores.size >= 2) scores[1] else 0f
            
            return DetectionResult(
                isEmergency = emergencyProbability > CONFIDENCE_THRESHOLD,
                confidence = emergencyProbability
            )
        } catch (e: Exception) {
            Log.e(TAG, "프레임 분석 오류: ${e.message}", e)
            return DetectionResult(
                isEmergency = false,
                confidence = 0f
            )
        }
    }
    
    data class DetectionResult(
        val isEmergency: Boolean,
        val confidence: Float
    )
    
    data class EmergencyAnalysisResult(
        val isEmergency: Boolean,
        val emergencyFrameRatio: Float,
        val maxConfidence: Float
    )
    
    companion object {
        private const val TAG = "EmergencyDetector"
        private const val CONFIDENCE_THRESHOLD = 0.6f // 응급 상황 판단을 위한 신뢰도 임계값
        private const val EMERGENCY_RATIO_THRESHOLD = 0.2f // 전체 프레임 중 응급 상황으로 판단된 프레임의 비율 임계값
    }
} 
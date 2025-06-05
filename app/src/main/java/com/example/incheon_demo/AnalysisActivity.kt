package com.example.incheon_demo

import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.incheon_demo.databinding.ActivityAnalysisBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class AnalysisActivity : AppCompatActivity() {
    
    companion object {
        const val EXTRA_VIDEO_PATH = "video_path"
        private const val TAG = "AnalysisActivity"
    }
    
    private lateinit var binding: ActivityAnalysisBinding
    private var videoPath: String? = null
    private var actionClassifier: ActionClassifier? = null
    private var analysisResult: ActionClassifier.ActionAnalysisResult? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        videoPath = intent.getStringExtra(EXTRA_VIDEO_PATH)
        if (videoPath == null) {
            Toast.makeText(this, "비디오 경로를 찾을 수 없습니다", Toast.LENGTH_SHORT).show()
            finish()
            return
        }
        
        // 동작 분류기 초기화
        actionClassifier = ActionClassifier(this)
        
        setupUI()
        startAnalysis()
    }
    
    private fun setupUI() {
        binding.btnRetry.setOnClickListener {
            startAnalysis()
        }
        
        binding.btnFinish.setOnClickListener {
            finish()
        }
        
        // 비디오 미리보기 설정
        videoPath?.let { path ->
            try {
                Log.d(TAG, "🎬 비디오 미리보기 설정 시작: $path")
                
                // 파일 존재 확인
                val videoFile = File(path)
                if (!videoFile.exists()) {
                    Log.e(TAG, "❌ 비디오 파일이 존재하지 않음: $path")
                    return@let
                }
                
                Log.d(TAG, "📄 비디오 파일 정보:")
                Log.d(TAG, "   - 경로: ${videoFile.absolutePath}")
                Log.d(TAG, "   - 크기: ${videoFile.length()} bytes")
                Log.d(TAG, "   - 존재: ${videoFile.exists()}")
                
                // 파일 URI 생성 (file:// 접두사 추가)
                val videoUri = Uri.fromFile(videoFile)
                Log.d(TAG, "🔗 생성된 URI: $videoUri")
                
                binding.videoView.setVideoURI(videoUri)
                binding.videoView.setOnPreparedListener { mediaPlayer ->
                    Log.d(TAG, "✅ 비디오 준비 완료")
                    mediaPlayer.isLooping = true
                    binding.videoView.start()
                    
                    // 비디오 크기 조정
                    binding.videoView.layoutParams.height = 400 // dp를 px로 변환 필요시
                    binding.videoView.requestLayout()
                }
                
                binding.videoView.setOnErrorListener { _, what, extra ->
                    Log.e(TAG, "❌ 비디오 재생 오류: what=$what, extra=$extra")
                    binding.videoView.visibility = View.GONE
                    true // 에러 처리 완료
                }
                
                binding.videoView.setOnInfoListener { _, what, extra ->
                    Log.d(TAG, "ℹ️ 비디오 정보: what=$what, extra=$extra")
                    false
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "💥 비디오 미리보기 설정 실패: ${e.message}", e)
                // 비디오뷰 숨기기
                binding.videoView.visibility = View.GONE
            }
        }
    }
    
    private fun startAnalysis() {
        binding.progressBar.visibility = View.VISIBLE
        //binding.statusText.text = "분석 준비 중..."
        binding.resultContainer.visibility = View.GONE
        binding.top3Container.visibility = View.GONE
        binding.btnRetry.visibility = View.GONE
        
        lifecycleScope.launch {
            try {
                analysisResult = analyzeVideo()
                showAnalysisResult(analysisResult!!)
            } catch (e: Exception) {
                Log.e(TAG, "분석 실패: ${e.message}", e)
                showError("분석 중 오류가 발생했습니다: ${e.message}")
            }
        }
    }
    
    private suspend fun analyzeVideo(): ActionClassifier.ActionAnalysisResult = withContext(Dispatchers.IO) {
        actionClassifier?.analyzeVideoWithProgress(videoPath!!, object : ActionClassifier.AnalysisProgressCallback {
            override fun onProgressUpdate(progress: Int, status: String) {
                lifecycleScope.launch {
                    binding.progressBar.progress = progress
                    binding.statusText.text = status
                }
            }
        }) ?: throw RuntimeException("ActionClassifier가 초기화되지 않음")
    }
    
    private fun showAnalysisResult(result: ActionClassifier.ActionAnalysisResult) {
        binding.progressBar.visibility = View.GONE
        binding.resultContainer.visibility = View.VISIBLE
        binding.top3Container.visibility = View.VISIBLE
        
        // 기본 결과 표시
        //binding.statusText.text = "분석 완료"
        binding.resultText.text = buildString {
            appendLine("🎬 총 세그먼트: ${result.totalSegments}개")
            appendLine("📊 응급 세그먼트: ${result.emergencySegments}개")
            appendLine("🎯 주요 동작: ${result.dominantAction}")
            appendLine("📈 신뢰도: ${String.format("%.1f", result.confidence * 100)}%")
        }
        
        // 응급상황 여부에 따른 색상 설정
        val backgroundColor = if (result.isEmergency) {
            Color.parseColor("#FFEBEE") // 연한 빨강
        } else {
            Color.parseColor("#E8F5E8") // 연한 초록
        }
        binding.resultContainer.setBackgroundColor(backgroundColor)
        
        // Top3 예측 결과 표시
        showTop3Predictions(result.topPredictions)
        
        binding.btnRetry.visibility = View.VISIBLE
        
        Log.d(TAG, "분석 결과 표시 완료")
    }
    
    private fun showTop3Predictions(predictions: List<ActionClassifier.ClassPrediction>) {
        // 기존 버튼들 제거
        binding.top3Container.removeAllViews()
        
        // 제목 추가
        val titleText = TextView(this).apply {
            text = "🏆 Top 3 예측 결과 (선택해주세요)"
            textSize = 18f
            setPadding(16, 16, 16, 8)
            setTextColor(Color.BLACK)
        }
        binding.top3Container.addView(titleText)
        
        // 설명 추가
        val descText = TextView(this).apply {
            text = "AI가 분석한 결과 중 가장 정확하다고 생각되는 동작을 선택해주세요"
            textSize = 14f
            setPadding(16, 0, 16, 16)
            setTextColor(Color.GRAY)
        }
        binding.top3Container.addView(descText)
        
        // Top3 버튼들 생성
        predictions.forEachIndexed { index, prediction ->
            val button = Button(this).apply {
                text = buildString {
                    append("${index + 1}위: ${prediction.className}")
                    appendLine()
                    append("신뢰도: ${String.format("%.1f", prediction.confidence * 100)}%")
                    if (prediction.isEmergency) {
                        append(" 🚨")
                    }
                }
                
                // 버튼 스타일 설정
                val layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    setMargins(16, 8, 16, 8)
                }
                this.layoutParams = layoutParams
                
                // 색상 설정
                val buttonColor = when {
                    prediction.isEmergency -> Color.parseColor("#FF5722") // 주황/빨강
                    index == 0 -> Color.parseColor("#4CAF50") // 초록 (1위)
                    else -> Color.parseColor("#2196F3") // 파랑 (2,3위)
                }
                setBackgroundColor(buttonColor)
                setTextColor(Color.WHITE)
                
                // 클릭 이벤트
                setOnClickListener {
                    confirmSelection(prediction, index + 1)
                }
            }
            
            binding.top3Container.addView(button)
        }
        
        // "모든 결과가 틀림" 버튼 추가
        val wrongButton = Button(this).apply {
            text = "❌ 모든 결과가 정확하지 않음"
            val layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(16, 16, 16, 8)
            }
            this.layoutParams = layoutParams
            setBackgroundColor(Color.parseColor("#607D8B")) // 회색
            setTextColor(Color.WHITE)
            
            setOnClickListener {
                showManualInputDialog()
            }
        }
        binding.top3Container.addView(wrongButton)
    }
    
    private fun confirmSelection(prediction: ActionClassifier.ClassPrediction, rank: Int) {
        val message = buildString {
            appendLine("선택하신 결과:")
            appendLine("")
            appendLine("🏆 순위: ${rank}위")
            appendLine("🎭 동작: ${prediction.className}")
            appendLine("📊 신뢰도: ${String.format("%.1f", prediction.confidence * 100)}%")
            appendLine("🚨 응급상황: ${if (prediction.isEmergency) "예" else "아니오"}")
            appendLine("")
            appendLine("이 결과가 정확합니까?")
        }
        
        AlertDialog.Builder(this)
            .setTitle("결과 확인")
            .setMessage(message)
            .setPositiveButton("네, 정확합니다") { _, _ ->
                saveFinalResult(prediction, "user_selected_rank_$rank")
            }
            .setNegativeButton("다시 선택") { _, _ ->
                // 다시 선택할 수 있도록 아무것도 하지 않음
            }
            .show()
    }
    
    private fun showManualInputDialog() {
        val actions = arrayOf(
            "걷기", "달리기", "점프", "앉기", 
            "일어서기", "넘어짐", "정상활동", "기타"
        )
        
        AlertDialog.Builder(this)
            .setTitle("정확한 동작 선택")
            .setItems(actions) { _, which ->
                val selectedAction = actions[which]
                val isEmergency = selectedAction == "넘어짐"
                
                val manualPrediction = ActionClassifier.ClassPrediction(
                    classIndex = which,
                    className = selectedAction,
                    confidence = 1.0f, // 사용자 선택이므로 100% 신뢰도
                    isEmergency = isEmergency
                )
                
                saveFinalResult(manualPrediction, "user_manual_input")
            }
            .setNegativeButton("취소", null)
            .show()
    }
    
    private fun saveFinalResult(finalPrediction: ActionClassifier.ClassPrediction, selectionMethod: String) {
        // 결과 로깅 (추후 모델 개선에 활용)
        Log.i(TAG, "=== 최종 사용자 선택 결과 ===")
        Log.i(TAG, "선택 방법: $selectionMethod")
        Log.i(TAG, "최종 동작: ${finalPrediction.className}")
        Log.i(TAG, "응급상황: ${finalPrediction.isEmergency}")
        Log.i(TAG, "신뢰도: ${finalPrediction.confidence}")
        Log.i(TAG, "비디오 경로: $videoPath")
        
        // TODO: 서버로 피드백 데이터 전송 (모델 개선용)
        
        // 성공 메시지 표시
        val message = if (finalPrediction.isEmergency) {
            "🚨 응급상황이 감지되었습니다!\n동작: ${finalPrediction.className}\n\n관련 기관에 신고하시겠습니까?"
        } else {
            "✅ 정상적인 활동으로 분류되었습니다.\n동작: ${finalPrediction.className}"
        }
        
        val builder = AlertDialog.Builder(this)
            .setTitle("최종 결과")
            .setMessage(message)
            .setPositiveButton("확인") { _, _ ->
                finish()
            }
        
        if (finalPrediction.isEmergency) {
            builder.setNeutralButton("신고하기") { _, _ ->
                // TODO: 응급신고 기능 구현
                showEmergencyOptions()
            }
        }
        
        builder.show()
    }
    
    private fun showEmergencyOptions() {
        val options = arrayOf(
            "119 (소방서)",
            "112 (경찰서)", 
            "관리사무소",
            "보호자 연락"
        )
        
        AlertDialog.Builder(this)
            .setTitle("응급신고")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> makeEmergencyCall("119")
                    1 -> makeEmergencyCall("112")
                    2 -> Toast.makeText(this, "관리사무소 연락 기능 준비 중", Toast.LENGTH_SHORT).show()
                    3 -> Toast.makeText(this, "보호자 연락 기능 준비 중", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("취소") { _, _ ->
                finish()
            }
            .show()
    }
    
    private fun makeEmergencyCall(number: String) {
        try {
            val intent = Intent(Intent.ACTION_CALL).apply {
                data = Uri.parse("tel:$number")
            }
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "전화 걸기 실패: ${e.message}")
            Toast.makeText(this, "전화 걸기 실패: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun showError(message: String) {
        binding.progressBar.visibility = View.GONE
        binding.statusText.text = "분석 실패"
        binding.resultText.text = message
        binding.resultContainer.visibility = View.VISIBLE
        binding.resultContainer.setBackgroundColor(Color.parseColor("#FFCDD2")) // 연한 빨강
        binding.btnRetry.visibility = View.VISIBLE
        binding.top3Container.visibility = View.GONE
    }
    
    override fun onDestroy() {
        super.onDestroy()
        binding.videoView.stopPlayback()
        actionClassifier?.cleanup()
    }
} 
package com.example.incheon_demo

import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.incheon_demo.databinding.ActivityAnalysisBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AnalysisActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityAnalysisBinding
    private var emergencyDetector: EmergencyDetector? = null
    private var videoPath: String? = null
    
    companion object {
        private const val TAG = "AnalysisActivity"
        const val EXTRA_VIDEO_PATH = "extra_video_path"
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // 비디오 경로 가져오기
        videoPath = intent.getStringExtra(EXTRA_VIDEO_PATH)
        
        if (videoPath == null) {
            Log.e(TAG, "비디오 경로가 전달되지 않았습니다")
            Toast.makeText(this, "비디오 파일을 찾을 수 없습니다", Toast.LENGTH_SHORT).show()
            finish()
            return
        }
        
        // EmergencyDetector 초기화
        emergencyDetector = EmergencyDetector(this)
        
        // 초기 UI 설정
        setupUI()
        
        // 분석 시작
        startAnalysis()
    }
    
    private fun setupUI() {
        // 초기 상태 설정
        binding.progressBar.progress = 0
        binding.tvAnalysisStatus.text = "분석을 준비하고 있습니다..."
        binding.tvProgressText.text = "0%"
        
        // 취소 버튼 클릭 리스너
        binding.btnCancel.setOnClickListener {
            showCancelDialog()
        }
    }
    
    private fun startAnalysis() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = emergencyDetector?.detectFromVideoWithProgress(
                    videoPath!!,
                    object : EmergencyDetector.AnalysisProgressCallback {
                        override fun onProgressUpdate(progress: Int, status: String) {
                            runOnUiThread {
                                updateProgress(progress, status)
                            }
                        }
                    }
                )
                
                // 메인 스레드에서 결과 처리
                withContext(Dispatchers.Main) {
                    handleAnalysisResult(result)
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Log.e(TAG, "분석 중 오류: ${e.message}", e)
                    showErrorDialog("분석 중 오류가 발생했습니다: ${e.message}")
                }
            }
        }
    }
    
    private fun updateProgress(progress: Int, status: String) {
        binding.progressBar.progress = progress
        binding.tvAnalysisStatus.text = status
        binding.tvProgressText.text = "${progress}%"
        
        // 진행 상황에 따른 추가 UI 업데이트
        when {
            progress < 20 -> {
                binding.ivAnalysisIcon.setImageResource(R.drawable.ic_video_file)
            }
            progress < 50 -> {
                binding.ivAnalysisIcon.setImageResource(R.drawable.ic_frame_extract)
            }
            progress < 90 -> {
                binding.ivAnalysisIcon.setImageResource(R.drawable.ic_ai_analysis)
            }
            else -> {
                binding.ivAnalysisIcon.setImageResource(R.drawable.ic_analysis_complete)
            }
        }
    }
    
    private fun handleAnalysisResult(result: EmergencyDetector.EmergencyAnalysisResult?) {
        if (result == null) {
            showErrorDialog("분석 결과를 가져올 수 없습니다")
            return
        }
        
        if (result.isEmergency) {
            showEmergencyResult(result)
        } else {
            showNormalResult(result)
        }
    }
    
    private fun showEmergencyResult(result: EmergencyDetector.EmergencyAnalysisResult) {
        AlertDialog.Builder(this)
            .setTitle("🚨 응급상황 감지!")
            .setMessage(
                "AI가 응급상황을 감지했습니다!\n\n" +
                "• 최대 신뢰도: ${String.format("%.1f", result.maxConfidence * 100)}%\n" +
                "• 응급 프레임 비율: ${String.format("%.1f", result.emergencyFrameRatio * 100)}%\n" +
                "• 분석된 프레임: ${result.totalFrames}개\n" +
                "• 주요 감지 클래스: ${result.dominantClass}\n\n" +
                "🤖 모델 상태: ${if (result.maxConfidence > 0.5f) "실제 AI 추론" else "⚠️ 테스트 모드 (낮은 신뢰도)"}\n\n" +
                "즉시 112에 신고하시겠습니까?"
            )
            .setPositiveButton("112 신고하기") { _, _ ->
                make112Call()
                finishWithResult(true, result)
            }
            .setNegativeButton("나중에") { _, _ ->
                finishWithResult(true, result)
            }
            .setCancelable(false)
            .show()
    }
    
    private fun showNormalResult(result: EmergencyDetector.EmergencyAnalysisResult) {
        AlertDialog.Builder(this)
            .setTitle("✅ 정상 상황")
            .setMessage(
                "응급상황이 감지되지 않았습니다.\n\n" +
                "• 최대 신뢰도: ${String.format("%.1f", result.maxConfidence * 100)}%\n" +
                "• 응급 프레임 비율: ${String.format("%.1f", result.emergencyFrameRatio * 100)}%\n" +
                "• 분석된 프레임: ${result.totalFrames}개\n" +
                "• 주요 감지 클래스: ${result.dominantClass}\n\n" +
                "🤖 모델 상태: ${if (result.maxConfidence > 0.5f) "실제 AI 추론" else "⚠️ 테스트 모드 (낮은 신뢰도)"}\n\n" +
                "영상이 안전하게 저장되었습니다."
            )
            .setPositiveButton("확인") { _, _ ->
                finishWithResult(false, result)
            }
            .setCancelable(true)
            .show()
    }
    
    private fun showCancelDialog() {
        AlertDialog.Builder(this)
            .setTitle("분석 취소")
            .setMessage("분석을 취소하시겠습니까?\n진행 중인 작업이 중단됩니다.")
            .setPositiveButton("취소하기") { _, _ ->
                finish()
            }
            .setNegativeButton("계속하기", null)
            .show()
    }
    
    private fun showErrorDialog(message: String) {
        AlertDialog.Builder(this)
            .setTitle("오류 발생")
            .setMessage(message)
            .setPositiveButton("확인") { _, _ ->
                finish()
            }
            .setCancelable(false)
            .show()
    }
    
    private fun make112Call() {
        try {
            val intent = Intent(Intent.ACTION_DIAL).apply {
                data = Uri.parse("tel:112")
            }
            startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "112 신고 중 오류: ${e.message}")
            Toast.makeText(this, "112 신고 중 오류가 발생했습니다", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun finishWithResult(isEmergency: Boolean, result: EmergencyDetector.EmergencyAnalysisResult) {
        val resultIntent = Intent().apply {
            putExtra("is_emergency", isEmergency)
            putExtra("max_confidence", result.maxConfidence)
            putExtra("emergency_frame_ratio", result.emergencyFrameRatio)
            putExtra("total_frames", result.totalFrames)
            putExtra("emergency_frames", result.emergencyFrames)
        }
        setResult(RESULT_OK, resultIntent)
        finish()
    }
    
    override fun onDestroy() {
        emergencyDetector?.cleanup()
        super.onDestroy()
    }
} 
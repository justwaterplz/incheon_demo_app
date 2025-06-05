package com.example.incheon_demo

import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.incheon_demo.databinding.ActivityTestAnalysisBinding

class TestAnalysisActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "TestAnalysisActivity"
        
        // 사용자가 제공한 6가지 테스트 케이스
        private val TEST_CASES = listOf(
            // Case 1: 싸움이 압도적으로 높음 (0.9995)
            TestCase(
                name = "Case 1 - 싸움 (99.95%)",
                scores = floatArrayOf(0.0000f, 0.0001f, 0.0001f, 0.0000f, 0.9995f, 0.0000f, 0.0000f, 0.0002f)
            ),
            // Case 2: 싸움이 높지만 다른 클래스도 어느정도 확률 (0.7173)
            TestCase(
                name = "Case 2 - 싸움 (71.73%)",
                scores = floatArrayOf(0.0394f, 0.1337f, 0.0214f, 0.0074f, 0.7173f, 0.0323f, 0.0139f, 0.0345f)
            ),
            // Case 3: 소매치기가 가장 높음 (0.3804)
            TestCase(
                name = "Case 3 - 소매치기 (38.04%)",
                scores = floatArrayOf(0.0747f, 0.3804f, 0.0040f, 0.0012f, 0.1236f, 0.1410f, 0.0068f, 0.2683f)
            ),
            // Case 4: 싸움이 높음 (0.9324)
            TestCase(
                name = "Case 4 - 싸움 (93.24%)",
                scores = floatArrayOf(0.0007f, 0.0624f, 0.0003f, 0.0000f, 0.9324f, 0.0013f, 0.0002f, 0.0027f)
            ),
            // Case 5: 싸움이 매우 높음 (0.9968)
            TestCase(
                name = "Case 5 - 싸움 (99.68%)",
                scores = floatArrayOf(0.0001f, 0.0019f, 0.0003f, 0.0000f, 0.9968f, 0.0003f, 0.0001f, 0.0004f)
            ),
            // Case 6: 소매치기가 가장 높음 (0.5307)
            TestCase(
                name = "Case 6 - 소매치기 (53.07%)",
                scores = floatArrayOf(0.0440f, 0.5307f, 0.0151f, 0.1189f, 0.0469f, 0.0534f, 0.0096f, 0.1814f)
            )
        )
        
        // AI 모델의 8개 클래스 레이블 (EmergencyDetector와 동일)
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
        
        // 응급상황으로 간주할 클래스 인덱스들 (정상(6)을 제외한 모든 클래스)
        private val EMERGENCY_CLASS_INDICES = setOf(0, 1, 2, 3, 4, 5, 7)
    }
    
    data class TestCase(
        val name: String,
        val scores: FloatArray
    )
    
    data class ClassPrediction(
        val classIndex: Int,
        val className: String,
        val confidence: Float,
        val isEmergency: Boolean
    )
    
    data class TestAnalysisResult(
        val isEmergency: Boolean,
        val topPredictions: List<ClassPrediction>,
        val confidence: Float,
        val totalSegments: Int,
        val emergencySegments: Int,
        val dominantAction: String
    )
    
    private lateinit var binding: ActivityTestAnalysisBinding
    private var currentCaseIndex = 0
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTestAnalysisBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupUI()
        showTestCase(currentCaseIndex)
    }
    
    private fun setupUI() {
        binding.btnRetry.text = "다음 케이스"
        binding.btnRetry.setOnClickListener {
            currentCaseIndex = (currentCaseIndex + 1) % TEST_CASES.size
            showTestCase(currentCaseIndex)
        }
        
        binding.btnFinish.setOnClickListener {
            finish()
        }
        
        // 비디오 뷰는 숨기기 (테스트용이므로)
        binding.videoView.visibility = View.GONE
    }
    
    private fun showTestCase(caseIndex: Int) {
        val testCase = TEST_CASES[caseIndex]
        
        // 프로그레스바 숨기기
        binding.progressBar.visibility = View.GONE
        binding.resultContainer.visibility = View.VISIBLE
        binding.top3Container.visibility = View.VISIBLE
        
        // 상태 텍스트 업데이트
        binding.statusText.text = "분석 완료"
        
        // 테스트 케이스를 분석 결과로 변환
        val analysisResult = convertToAnalysisResult(testCase)
        
        // 분석 결과 표시
        showAnalysisResult(analysisResult)
    }
    
    private fun convertToAnalysisResult(testCase: TestCase): TestAnalysisResult {
        // 각 클래스별 예측 결과 생성
        val predictions = testCase.scores.mapIndexed { index, score ->
            ClassPrediction(
                classIndex = index,
                className = CLASS_LABELS[index],
                confidence = score,
                isEmergency = EMERGENCY_CLASS_INDICES.contains(index)
            )
        }
        
        // Top3 예측 결과 (확률 순으로 정렬)
        val top3Predictions = predictions
            .sortedByDescending { it.confidence }
            .take(3)
        
        // 가장 높은 확률의 클래스
        val dominantPrediction = top3Predictions.first()
        
        // 응급상황 여부 판단
        val isEmergency = dominantPrediction.isEmergency && dominantPrediction.confidence > 0.5f
        
        // 가상의 세그먼트 정보 생성
        val totalSegments = 30  // 1분 영상, 2초 세그먼트 기준
        val emergencySegments = if (isEmergency) (totalSegments * 0.8).toInt() else 0
        
        return TestAnalysisResult(
            isEmergency = isEmergency,
            topPredictions = top3Predictions,
            confidence = dominantPrediction.confidence,
            totalSegments = totalSegments,
            emergencySegments = emergencySegments,
            dominantAction = dominantPrediction.className
        )
    }
    
    private fun showAnalysisResult(result: TestAnalysisResult) {
        // 기본 결과 표시
        binding.resultText.text = buildString {
            //appendLine("🎬 총 세그먼트: ${result.totalSegments}개")
            //appendLine("📊 응급 세그먼트: ${result.emergencySegments}개")
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
    }
    
    private fun showTop3Predictions(predictions: List<ClassPrediction>) {
        // 기존 뷰들 제거
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
                    append(" (${String.format("%.2f", prediction.confidence * 100)}%)")
                    if (prediction.isEmergency) append(" ⚠️")
                }
                
                setBackgroundColor(
                    when (index) {
                        0 -> Color.parseColor("#4CAF50") // 1위: 초록
                        1 -> Color.parseColor("#FF9800") // 2위: 주황
                        2 -> Color.parseColor("#9E9E9E") // 3위: 회색
                        else -> Color.parseColor("#E0E0E0")
                    }
                )
                setTextColor(Color.WHITE)
                setPadding(32, 24, 32, 24)
                
                val params = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply {
                    setMargins(16, 8, 16, 8)
                }
                layoutParams = params
                
                setOnClickListener {
                    showPredictionDetails(prediction, index + 1)
                }
            }
            binding.top3Container.addView(button)
        }
        
        // "모든 결과가 정확하지 않음" 버튼 추가
        val incorrectButton = Button(this).apply {
            text = "❌ 모든 결과가 정확하지 않음"
            setBackgroundColor(Color.parseColor("#F44336")) // 빨간색
            setTextColor(Color.WHITE)
            setPadding(32, 24, 32, 24)
            
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                setMargins(16, 16, 16, 8)
            }
            layoutParams = params
            
            setOnClickListener {
                showIncorrectResultDialog()
            }
        }
        binding.top3Container.addView(incorrectButton)
    }
    
    private fun showIncorrectResultDialog() {
        val message = buildString {
            appendLine("❌ 모든 결과가 정확하지 않음")
            appendLine("")
            appendLine("AI가 제시한 3가지 예측 결과가 모두 정확하지 않은 경우입니다.")
            appendLine("")
            appendLine("실제 시스템에서는 사용자가 올바른 동작을 직접 입력하거나")
            appendLine("추가적인 분석을 진행할 수 있습니다.")
        }
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("결과 선택")
            .setMessage(message)
            .setPositiveButton("확인", null)
            .show()
    }
    
    private fun showPredictionDetails(prediction: ClassPrediction, rank: Int) {
        val message = buildString {
            appendLine("🏆 ${rank}위 예측 결과")
            appendLine("")
            appendLine("🎯 클래스: ${prediction.className} (${prediction.classIndex})")
            appendLine("📈 신뢰도: ${String.format("%.4f", prediction.confidence)} (${String.format("%.2f", prediction.confidence * 100)}%)")
            appendLine("⚠️ 응급상황: ${if (prediction.isEmergency) "예" else "아니오"}")
            appendLine("")
            appendLine("이것은 PyTorch 모델의 실제 출력값을 기반으로 한 테스트용 화면입니다.")
        }
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("예측 결과 상세")
            .setMessage(message)
            .setPositiveButton("확인", null)
            .show()
    }
} 
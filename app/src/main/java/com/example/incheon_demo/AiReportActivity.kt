package com.example.incheon_demo

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.button.MaterialButton

class AiReportActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ai_report)

        // 툴바 설정
        val toolbar = findViewById<MaterialToolbar>(R.id.toolbar)
        toolbar.setNavigationOnClickListener {
            finish()
        }

        // 신고하기 버튼 클릭 리스너
        findViewById<MaterialButton>(R.id.btnSubmit).setOnClickListener {
            // TODO: AI 모델을 사용한 신고 처리 구현
            Toast.makeText(this, "신고가 접수되었습니다.", Toast.LENGTH_SHORT).show()
            finish()
        }
    }
} 
package com.example.incheon_demo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.card.MaterialCardView
import com.permissionx.guolindev.PermissionX
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_SELECT_VIDEO = 1001
        private const val REQUEST_CODE_ANALYSIS = 1002
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        checkAndRequestPermissions()

        // 기존 버튼들
        findViewById<MaterialCardView>(R.id.btnSmartAiReport).setOnClickListener {
            startActivity(Intent(this, CameraActivity::class.java))
        }
        
        // 테스트용 비디오 파일 분석 버튼
        findViewById<MaterialCardView>(R.id.btnVideoFileTest).setOnClickListener {
            selectVideoFile()
        }
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.CAMERA,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.SEND_SMS
        ).apply {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.READ_MEDIA_VIDEO)
            } else {
                add(Manifest.permission.READ_EXTERNAL_STORAGE)
                add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
        }

        // 승인되지 않은 권한만 필터링
        val notGrantedPermissions = permissions.filter { permission ->
            ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED
        }

        if (notGrantedPermissions.isNotEmpty()) {
            PermissionX.init(this)
                .permissions(notGrantedPermissions)
                .request { allGranted, _, deniedList ->
                    if (allGranted) {
                        Toast.makeText(this, "모든 권한이 승인되었습니다.", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this, "일부 권한이 거부되었습니다: $deniedList", Toast.LENGTH_LONG).show()
                    }
                }
        }
    }
    
    private fun selectVideoFile() {
        try {
            val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                type = "video/*"
                addCategory(Intent.CATEGORY_OPENABLE)
                putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("video/mp4", "video/avi", "video/mov", "video/3gp"))
            }
            
            if (intent.resolveActivity(packageManager) != null) {
                startActivityForResult(
                    Intent.createChooser(intent, "테스트할 비디오 파일 선택"),
                    REQUEST_CODE_SELECT_VIDEO
                )
            } else {
                Toast.makeText(this, "파일 선택 앱을 찾을 수 없습니다", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "비디오 파일 선택 중 오류: ${e.message}", e)
            Toast.makeText(this, "파일 선택 중 오류가 발생했습니다", Toast.LENGTH_SHORT).show()
        }
    }
    
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        when (requestCode) {
            REQUEST_CODE_SELECT_VIDEO -> {
                if (resultCode == RESULT_OK && data?.data != null) {
                    handleSelectedVideo(data.data!!)
                } else {
                    Log.d(TAG, "비디오 파일 선택이 취소되었습니다")
                }
            }
            REQUEST_CODE_ANALYSIS -> {
                if (resultCode == RESULT_OK && data != null) {
                    val isEmergency = data.getBooleanExtra("is_emergency", false)
                    val maxConfidence = data.getFloatExtra("max_confidence", 0f)
                    val emergencyFrameRatio = data.getFloatExtra("emergency_frame_ratio", 0f)
                    val totalFrames = data.getIntExtra("total_frames", 0)
                    val emergencyFrames = data.getIntExtra("emergency_frames", 0)
                    
                    Log.d(TAG, "📊 테스트 분석 결과:")
                    Log.d(TAG, "   - 응급상황: $isEmergency")
                    Log.d(TAG, "   - 최고 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%")
                    Log.d(TAG, "   - 응급 프레임 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%")
                    Log.d(TAG, "   - 총 프레임: $totalFrames, 응급 프레임: $emergencyFrames")
                    
                    if (isEmergency) {
                        Toast.makeText(this, 
                            "🚨 테스트 결과: 응급상황 감지!\n" +
                            "📊 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%\n" +
                            "📈 응급 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%\n" +
                            "🎯 기준: 비율>50% & 신뢰도>80%", 
                            Toast.LENGTH_LONG).show()
                    } else {
                        Toast.makeText(this, 
                            "✅ 테스트 결과: 정상 상황\n" +
                            "📊 최고 신뢰도: ${String.format("%.1f", maxConfidence * 100)}%\n" +
                            "📈 응급 비율: ${String.format("%.1f", emergencyFrameRatio * 100)}%\n" +
                            "💡 현재 기준: 매우 엄격한 모드", 
                            Toast.LENGTH_LONG).show()
                    }
                } else {
                    Log.d(TAG, "테스트 분석이 취소되었거나 실패했습니다")
                    Toast.makeText(this, "테스트가 취소되었습니다", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }
    
    private fun handleSelectedVideo(videoUri: Uri) {
        try {
            Log.d(TAG, "선택된 비디오: $videoUri")
            
            // URI를 실제 파일 경로로 변환 (AnalysisActivity에서 사용할 수 있도록)
            val videoPath = copyUriToTempFile(videoUri)
            
            if (videoPath != null) {
                Log.d(TAG, "비디오 파일 복사 완료: $videoPath")
                
                // AnalysisActivity로 이동
                val intent = Intent(this, AnalysisActivity::class.java).apply {
                    putExtra(AnalysisActivity.EXTRA_VIDEO_PATH, videoPath)
                }
                startActivityForResult(intent, REQUEST_CODE_ANALYSIS)
                
            } else {
                Toast.makeText(this, "비디오 파일을 처리하는 중 오류가 발생했습니다", Toast.LENGTH_SHORT).show()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "비디오 처리 중 오류: ${e.message}", e)
            Toast.makeText(this, "비디오 처리 중 오류가 발생했습니다: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun copyUriToTempFile(uri: Uri): String? {
        return try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            
            if (inputStream == null) {
                Log.e(TAG, "InputStream이 null입니다")
                return null
            }
            
            // 임시 파일 생성
            val tempFile = File(cacheDir, "temp_video_${System.currentTimeMillis()}.mp4")
            val outputStream = FileOutputStream(tempFile)
            
            // 파일 복사
            val buffer = ByteArray(8192)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }
            
            inputStream.close()
            outputStream.close()
            
            Log.d(TAG, "임시 파일 생성 완료: ${tempFile.absolutePath} (크기: ${tempFile.length()} bytes)")
            
            tempFile.absolutePath
            
        } catch (e: Exception) {
            Log.e(TAG, "파일 복사 중 오류: ${e.message}", e)
            null
        }
    }
} 
package com.example.incheon_demo

import android.Manifest
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import androidx.core.util.Consumer
import androidx.lifecycle.lifecycleScope
import com.example.incheon_demo.databinding.ActivityCameraBinding
import com.permissionx.guolindev.PermissionX
import kotlinx.coroutines.Job
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Locale
import androidx.camera.view.PreviewView
import com.google.android.material.floatingactionbutton.FloatingActionButton
import androidx.core.app.ActivityCompat

class CameraActivity : AppCompatActivity() {
    private lateinit var binding: ActivityCameraBinding
    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null
    private var videoPath: String? = null
    private var timerJob: Job? = null
    private var isRecording = false
    private var emergencyDetector: EmergencyDetector? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        try {
            // EmergencyDetector 초기화 (모델이 없어도 오류 없이 진행)
            emergencyDetector = EmergencyDetector(this)
            Log.d(TAG, "CameraActivity onCreate 시작")
            requestPermissions()
        } catch (e: Exception) {
            Log.e(TAG, "onCreate 에러: ${e.message}", e)
            Toast.makeText(this, "초기화 중 오류가 발생했습니다: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
        }

        binding.btnStartRecording.setOnClickListener {
            if (!isRecording) {
                startRecording()
            } else {
                stopRecording()
            }
        }
    }

    private fun requestPermissions() {
        Log.d(TAG, "권한 요청 시작")
        PermissionX.init(this)
            .permissions(Manifest.permission.CAMERA)
            .request { allGranted, _, deniedList ->
                Log.d(TAG, "권한 요청 결과: allGranted=$allGranted, deniedList=$deniedList")
                if (allGranted) {
                    startCamera()
                } else {
                    Toast.makeText(this, "카메라 권한이 필요합니다: $deniedList", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }
    }

    private fun startCamera() {
        Log.d(TAG, "카메라 시작")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                Log.d(TAG, "카메라 프로바이더 설정 시작")
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                    }

                val recorder = Recorder.Builder()
                    .setQualitySelector(QualitySelector.from(Quality.HD))
                    .build()
                videoCapture = VideoCapture.withOutput(recorder)

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    videoCapture
                )
                Log.d(TAG, "카메라 바인딩 성공")
            } catch (e: Exception) {
                Log.e(TAG, "카메라 바인딩 실패", e)
                Toast.makeText(this, "카메라 초기화 실패: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun startRecording() {
        val videoCapture = this.videoCapture ?: return
        binding.btnStartRecording.isEnabled = false

        val name = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss-SSS", Locale.getDefault())
            .format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
        }

        val mediaStoreOutputOptions = MediaStoreOutputOptions
            .Builder(contentResolver, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            .setContentValues(contentValues)
            .build()

        recording = videoCapture.output
            .prepareRecording(this, mediaStoreOutputOptions)
            .start(ContextCompat.getMainExecutor(this), recordingListener)

        isRecording = true
        binding.btnStartRecording.icon = ContextCompat.getDrawable(this, R.drawable.ic_stop)
        binding.btnStartRecording.isEnabled = true
        
        // 1분 타이머 시작
        startTimer()
    }

    private fun stopRecording() {
        val recording = recording
        if (recording != null) {
            isRecording = false
            timerJob?.cancel()
            binding.timerText.visibility = View.GONE
            binding.btnStartRecording.isEnabled = false
            recording.stop()
            this.recording = null
        }
    }

    private val recordingListener = Consumer<VideoRecordEvent> { event ->
        when (event) {
            is VideoRecordEvent.Start -> {
                binding.timerText.visibility = View.VISIBLE
            }
            is VideoRecordEvent.Finalize -> {
                if (event.hasError()) {
                    recording?.close()
                    recording = null
                    Log.e(TAG, "Video 녹화 에러: ${event.error}")
                    Toast.makeText(this, "녹화 중 오류가 발생했습니다.", Toast.LENGTH_SHORT).show()
                } else {
                    videoPath = event.outputResults.outputUri.toString()
                    Log.d(TAG, "녹화 완료: $videoPath")
                    
                    // 카메라 리소스를 먼저 정리한 후 분석 화면으로 이동
                    lifecycleScope.launch {
                        try {
                            // 카메라 즉시 정리
                            cleanupCameraResources()
                            
                            // 리소스 정리를 위한 짧은 지연
                            kotlinx.coroutines.delay(200)
                            
                            // 분석 화면으로 이동
                            val intent = Intent(this@CameraActivity, AnalysisActivity::class.java).apply {
                                putExtra(AnalysisActivity.EXTRA_VIDEO_PATH, videoPath)
                            }
                            startActivityForResult(intent, REQUEST_CODE_ANALYSIS)
                            
                        } catch (e: Exception) {
                            Log.e(TAG, "화면 전환 중 오류: ${e.message}")
                            Toast.makeText(this@CameraActivity, "화면 전환 중 오류가 발생했습니다", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
                binding.btnStartRecording.isEnabled = true
                binding.timerText.visibility = View.GONE
                isRecording = false
            }
        }
    }

    private fun startTimer() {
        var remainingSeconds = 60 // 최대 1분
        timerJob?.cancel()
        timerJob = lifecycleScope.launch {
            while (remainingSeconds >= 0 && isRecording) {
                withContext(Dispatchers.Main) {
                    binding.timerText.text = String.format("%02d:%02d", remainingSeconds / 60, remainingSeconds % 60)
                }
                delay(1000)
                remainingSeconds--
                
                if (remainingSeconds < 0) {
                    stopRecording()
                }
            }
        }
    }

    override fun onDestroy() {
        try {
            // 녹화가 진행 중이면 중지
            recording?.stop()
            
            // CameraX 정리
            videoCapture?.let { videoCapture ->
                val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()
                    cameraProvider.unbindAll()
                }, ContextCompat.getMainExecutor(this))
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "onDestroy 중 오류: ${e.message}")
        } finally {
            super.onDestroy()
        }
    }

    override fun onPause() {
        super.onPause()
        // 화면이 보이지 않을 때 즉시 카메라 정리
        if (!isRecording) {
            cleanupCameraResources()
        }
    }

    override fun onStop() {
        super.onStop()
        // Activity가 완전히 멈출 때 강제 정리
        cleanupCameraResources()
    }

    // 112 신고 기능 추가
    private fun make112Call() {
        try {
            val intent = Intent(Intent.ACTION_CALL).apply {
                data = Uri.parse("tel:112")
            }
            
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) 
                == PackageManager.PERMISSION_GRANTED) {
                startActivity(intent)
            } else {
                // 전화 권한이 없으면 다이얼 화면만 열기
                val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                    data = Uri.parse("tel:112")
                }
                startActivity(dialIntent)
            }
        } catch (e: Exception) {
            Log.e(TAG, "112 신고 중 오류: ${e.message}")
            Toast.makeText(this, "112 신고 중 오류가 발생했습니다", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        
        if (requestCode == REQUEST_CODE_ANALYSIS) {
            if (resultCode == RESULT_OK && data != null) {
                val isEmergency = data.getBooleanExtra("is_emergency", false)
                val maxConfidence = data.getFloatExtra("max_confidence", 0f)
                val emergencyFrameRatio = data.getFloatExtra("emergency_frame_ratio", 0f)
                val totalFrames = data.getIntExtra("total_frames", 0)
                val emergencyFrames = data.getIntExtra("emergency_frames", 0)
                
                Log.d(TAG, "분석 결과: 응급상황=$isEmergency, 신뢰도=$maxConfidence")
                
                if (isEmergency) {
                    Toast.makeText(this, "응급상황이 감지되었습니다!", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "정상 상황입니다.", Toast.LENGTH_SHORT).show()
                }
            } else {
                Log.d(TAG, "분석이 취소되었거나 실패했습니다")
            }
            
            // 메인 화면으로 돌아가기
            finish()
        }
    }

    private fun cleanupCameraResources() {
        try {
            Log.d(TAG, "카메라 리소스 정리 시작")
            
            // 녹화 중단
            recording?.stop()
            recording = null
            
            // CameraX 즉시 해제
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
            if (cameraProviderFuture.isDone) {
                val cameraProvider = cameraProviderFuture.get()
                cameraProvider.unbindAll()
                Log.d(TAG, "CameraX 즉시 해제 완료")
            } else {
                cameraProviderFuture.addListener({
                    try {
                        val cameraProvider = cameraProviderFuture.get()
                        cameraProvider.unbindAll()
                        Log.d(TAG, "CameraX 비동기 해제 완료")
                    } catch (e: Exception) {
                        Log.w(TAG, "CameraX 해제 중 오류: ${e.message}")
                    }
                }, ContextCompat.getMainExecutor(this))
            }
            
            // VideoCapture 참조 정리
            videoCapture = null
            
        } catch (e: Exception) {
            Log.w(TAG, "카메라 리소스 정리 중 오류: ${e.message}")
        }
    }

    companion object {
        private const val TAG = "CameraActivity"
        const val EXTRA_VIDEO_URI = "extra_video_uri"
        const val EXTRA_EMERGENCY_DETECTED = "extra_emergency_detected"
        const val EXTRA_EMERGENCY_CONFIDENCE = "extra_emergency_confidence"
        const val EXTRA_EMERGENCY_FRAME_RATIO = "extra_emergency_frame_ratio"
        const val REQUEST_CODE_ANALYSIS = 1
    }
} 
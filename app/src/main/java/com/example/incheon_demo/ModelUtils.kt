package com.example.incheon_demo

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object ModelUtils {
    
    /**
     * Android assets에서 파일을 임시 디렉토리로 복사하고 경로를 반환
     * PyTorch Mobile에서 사용할 수 있도록 실제 파일 경로가 필요함
     */
    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        
        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
            return file.absolutePath
        } catch (e: IOException) {
            throw IOException("Error copying asset file $assetName: ${e.message}", e)
        }
    }
} 
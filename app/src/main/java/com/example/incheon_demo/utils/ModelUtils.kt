package com.example.incheon_demo.utils

import android.content.Context
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream

object ModelUtils {
    fun loadModelFromAssets(context: Context, modelName: String): Module {
        val modelPath = getModelPath(context, modelName)
        return LiteModuleLoader.load(modelPath)
    }

    private fun getModelPath(context: Context, modelName: String): String {
        val modelDir = File(context.filesDir, "models")
        if (!modelDir.exists()) {
            modelDir.mkdirs()
        }

        val modelFile = File(modelDir, modelName)
        if (!modelFile.exists()) {
            context.assets.open("models/$modelName").use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return modelFile.absolutePath
    }
    
    // 비트맵을 텐서로 변환하는 유틸리티 함수
    fun bitmapToFloat32Tensor(
        bitmap: android.graphics.Bitmap,
        mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
        std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
    ): org.pytorch.Tensor {
        return org.pytorch.torchvision.TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            mean,
            std
        )
    }
} 
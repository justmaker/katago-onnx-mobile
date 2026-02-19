package com.justmaker.katago_onnx_mobile

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import java.io.*

/**
 * KataGo Engine wrapper for Android (ONNX Backend, Single-threaded).
 * Uses synchronous JNI API - no pthread, no pipe.
 */
class KataGoEngine(private val context: Context) {
    // JNI Native Methods (synchronous)
    private external fun initializeNative(
        config: String,
        modelBin: String,
        modelOnnx: String,
        boardSize: Int
    ): Boolean

    private external fun analyzePositionNative(
        boardXSize: Int,
        boardYSize: Int,
        komi: Double,
        maxVisits: Int,
        moves: Array<Array<String>>
    ): String

    private external fun destroyNative()

    companion object {
        private const val TAG = "KataGoEngine"
        private const val MODEL_BIN_FILE = "model.bin.gz"

        var nativeLoaded = false
            private set
        var nativeLoadError: String? = null
            private set

        fun loadNativeLibrary() {
            if (nativeLoaded) return
            try {
                System.loadLibrary("katago_mobile")
                nativeLoaded = true
                Log.i(TAG, "Native library loaded (ONNX backend)")
            } catch (e: UnsatisfiedLinkError) {
                nativeLoadError = e.message
                Log.e(TAG, "Failed to load native library: ${e.message}")
            }
        }
    }

    private var isInitialized = false
    private var initializedBoardSize = 0

    suspend fun start(boardSize: Int = 19): Boolean = withContext(Dispatchers.IO) {
        if (isInitialized && initializedBoardSize == boardSize) return@withContext true

        if (isInitialized && initializedBoardSize != boardSize) {
            Log.i(TAG, "Board size changed: $initializedBoardSize -> $boardSize, reinitializing...")
            destroyNative()
            isInitialized = false
            initializedBoardSize = 0
        }

        if (!nativeLoaded) {
            Log.e(TAG, "Cannot start: native library not loaded ($nativeLoadError)")
            return@withContext false
        }

        try {
            val modelBinPath = extractAsset("katago/$MODEL_BIN_FILE")
                ?: return@withContext false
            val modelOnnxPath = extractAsset("katago/model.onnx")
                ?: return@withContext false
            val configPath = createConfigFile()

            Log.i(TAG, "Initializing KataGo (ONNX backend) for ${boardSize}x${boardSize}...")
            val success = initializeNative(configPath, modelBinPath, modelOnnxPath, boardSize)

            if (success) {
                isInitialized = true
                initializedBoardSize = boardSize
                Log.i(TAG, "KataGo initialized for ${boardSize}x${boardSize}")
                return@withContext true
            } else {
                Log.e(TAG, "KataGo initialization failed")
                return@withContext false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Initialization error", e)
            return@withContext false
        }
    }

    fun stop() {
        if (!isInitialized) return
        destroyNative()
        isInitialized = false
        Log.i(TAG, "KataGo destroyed")
    }

    suspend fun analyze(
        boardSize: Int,
        moves: List<String>,
        komi: Double,
        maxVisits: Int
    ): String = withContext(Dispatchers.IO) {
        if (!isInitialized || initializedBoardSize != boardSize) {
            val started = start(boardSize)
            if (!started) {
                return@withContext "{\"error\": \"Failed to initialize engine for ${boardSize}x${boardSize}\"}"
            }
        }

        try {
            Log.d(TAG, "Analyzing: ${boardSize}x${boardSize}, ${moves.size} moves, komi=$komi, visits=$maxVisits")

            val movesArray = moves.map { move ->
                val parts = move.trim().split(" ", limit = 2)
                if (parts.size == 2) {
                    arrayOf(parts[0], parts[1])
                } else {
                    arrayOf("B", "pass")
                }
            }.toTypedArray()

            val result = analyzePositionNative(
                boardSize, boardSize,
                komi, maxVisits,
                movesArray
            )

            Log.d(TAG, "Analysis completed: ${result.length} bytes")
            return@withContext result

        } catch (e: Exception) {
            Log.e(TAG, "Analysis exception", e)
            return@withContext "{\"error\": \"${e.message}\"}"
        }
    }

    fun isEngineRunning(): Boolean = isInitialized

    private fun extractAsset(assetPath: String): String? {
        val filename = assetPath.substringAfterLast('/')
        val outputFile = File(context.cacheDir, filename)

        if (outputFile.exists() && outputFile.length() > 0) {
            Log.d(TAG, "Asset cached: $assetPath")
            return outputFile.absolutePath
        }

        // Try multiple asset path variants (plugin assets have different prefix)
        val paths = listOf(
            "flutter_assets/packages/katago_onnx_mobile/assets/$assetPath",
            "flutter_assets/assets/$assetPath",
            "flutter_assets/$assetPath",
            assetPath
        )

        for (path in paths) {
            try {
                context.assets.open(path).use { input ->
                    FileOutputStream(outputFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Log.i(TAG, "Asset extracted: $path -> ${outputFile.absolutePath}")
                return outputFile.absolutePath
            } catch (e: Exception) {
                // Try next path
            }
        }

        Log.e(TAG, "Failed to extract asset: $assetPath")
        return null
    }

    private fun createConfigFile(): String {
        val configFile = File(context.cacheDir, "analysis.cfg")

        val config = """
            # KataGo Analysis Config for Android (Single-threaded)
            numSearchThreads = 1
            numAnalysisThreads = 1
            numNNServerThreadsPerModel = 1
            maxVisits = 100
            reportAnalysisWinratesAs = BLACK
            nnCacheSizePowerOfTwo = 18
            nnMutexPoolSizePowerOfTwo = 14
            nnMaxBatchSize = 1
            logSearchInfo = false
            logToStderr = false
        """.trimIndent()

        configFile.writeText(config)
        return configFile.absolutePath
    }
}

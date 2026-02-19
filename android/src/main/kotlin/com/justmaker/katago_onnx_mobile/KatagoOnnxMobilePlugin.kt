package com.justmaker.katago_onnx_mobile

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*

/** KatagoOnnxMobilePlugin - Flutter plugin for KataGo ONNX inference on Android. */
class KatagoOnnxMobilePlugin :
    FlutterPlugin,
    MethodCallHandler {

    companion object {
        private const val TAG = "KatagoOnnxMobilePlugin"
        private const val METHOD_CHANNEL = "com.justmaker.katago_onnx_mobile/engine"
        private const val EVENT_CHANNEL = "com.justmaker.katago_onnx_mobile/events"
    }

    private lateinit var channel: MethodChannel
    private lateinit var eventChannel: EventChannel
    private var eventSink: EventChannel.EventSink? = null
    private var context: Context? = null
    private var kataGoEngine: KataGoEngine? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())

    private val isEmulator by lazy {
        Build.FINGERPRINT.contains("generic") ||
            Build.FINGERPRINT.contains("emulator") ||
            Build.MODEL.contains("Emulator") ||
            Build.PRODUCT.contains("sdk") ||
            Build.HARDWARE.contains("ranchu")
    }

    private fun ensureKataGoEngine(): Boolean {
        if (kataGoEngine != null) return true
        val ctx = context ?: return false
        if (isEmulator) {
            Log.w(TAG, "Emulator detected, skipping KataGo native engine")
            return false
        }

        KataGoEngine.loadNativeLibrary()
        if (KataGoEngine.nativeLoaded) {
            kataGoEngine = KataGoEngine(ctx)
            return true
        } else {
            Log.w(TAG, "KataGo native library unavailable: ${KataGoEngine.nativeLoadError}")
            return false
        }
    }

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        context = flutterPluginBinding.applicationContext

        channel = MethodChannel(flutterPluginBinding.binaryMessenger, METHOD_CHANNEL)
        channel.setMethodCallHandler(this)

        eventChannel = EventChannel(flutterPluginBinding.binaryMessenger, EVENT_CHANNEL)
        eventChannel.setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
                    eventSink = events
                    Log.d(TAG, "Event channel listening")
                }
                override fun onCancel(arguments: Any?) {
                    eventSink = null
                    Log.d(TAG, "Event channel cancelled")
                }
            }
        )
    }

    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "startEngine" -> {
                scope.launch(Dispatchers.IO) {
                    val boardSize = call.argument<Int>("boardSize") ?: 19
                    val success = if (ensureKataGoEngine()) {
                        kataGoEngine?.start(boardSize) ?: false
                    } else {
                        false
                    }
                    withContext(Dispatchers.Main) {
                        result.success(success)
                    }
                }
            }

            "stopEngine" -> {
                kataGoEngine?.stop()
                result.success(true)
            }

            "isEngineRunning" -> {
                result.success(kataGoEngine?.isEngineRunning() ?: false)
            }

            "analyze" -> {
                scope.launch(Dispatchers.IO) {
                    if (kataGoEngine == null) {
                        ensureKataGoEngine()
                    }

                    val boardSize = call.argument<Int>("boardSize")
                        ?: call.argument<Int>("boardXSize") ?: 19
                    val moves = call.argument<List<String>>("moves") ?: emptyList()
                    val komi = call.argument<Double>("komi") ?: 7.5
                    val maxVisits = call.argument<Int>("maxVisits") ?: 100

                    val response = kataGoEngine?.analyze(
                        boardSize = boardSize,
                        moves = moves,
                        komi = komi,
                        maxVisits = maxVisits
                    ) ?: "{\"error\": \"Engine not available\"}"

                    mainHandler.post {
                        eventSink?.success(mapOf(
                            "type" to "analysis",
                            "data" to response
                        ))
                    }

                    withContext(Dispatchers.Main) {
                        result.success(response)
                    }
                }
            }

            "cancelAnalysis" -> {
                result.success(true)
            }

            else -> {
                result.notImplemented()
            }
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        kataGoEngine?.stop()
        scope.cancel()
        context = null
    }
}

import Flutter
import UIKit

// MARK: - ONNX Engine State
private var isOnnxInitialized = false

public class KatagoOnnxMobilePlugin: NSObject, FlutterPlugin, FlutterStreamHandler {

    private var eventSink: FlutterEventSink?
    private var progressTimer: DispatchSourceTimer?
    private let engineQueue = DispatchQueue(label: "com.justmaker.katago.engine")

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "com.justmaker.katago_onnx_mobile/engine",
            binaryMessenger: registrar.messenger()
        )
        let eventChannel = FlutterEventChannel(
            name: "com.justmaker.katago_onnx_mobile/events",
            binaryMessenger: registrar.messenger()
        )

        let instance = KatagoOnnxMobilePlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
        eventChannel.setStreamHandler(instance)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "startEngineOnnx":
            startEngineOnnx(args: call.arguments as? [String: Any], result: result)
        case "analyzeOnnx":
            analyzeOnnx(args: call.arguments as? [String: Any], result: result)
        case "cancelAnalysisOnnx":
            cancelAnalysisOnnx(result: result)
        case "stopEngine", "stopEngineOnnx":
            stopEngineOnnx(result: result)
        case "isEngineRunning", "isEngineOnnxRunning":
            result(isOnnxInitialized)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - FlutterStreamHandler

    public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        self.eventSink = events
        return nil
    }

    public func onCancel(withArguments arguments: Any?) -> FlutterError? {
        self.eventSink = nil
        return nil
    }

    // MARK: - ONNX Engine Methods

    private func startEngineOnnx(args: [String: Any]?, result: @escaping FlutterResult) {
        #if targetEnvironment(simulator)
        NSLog("[KataGoONNX] Simulator detected, skipping ONNX engine")
        result(false)
        return
        #endif

        guard let args = args else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing arguments", details: nil))
            return
        }

        let boardSize = args["boardSize"] as? Int ?? 19

        if isOnnxInitialized {
            NSLog("[KataGoONNX] Engine already initialized")
            result(true)
            return
        }

        DispatchQueue.global(qos: .userInitiated).async {
            let (configPath, modelBinPath, modelOnnxPath) = self.prepareOnnxResources(boardSize: boardSize)

            guard let config = configPath,
                  let modelBin = modelBinPath,
                  let modelOnnx = modelOnnxPath else {
                DispatchQueue.main.async {
                    result(FlutterError(code: "RESOURCE_ERROR", message: "Failed to prepare resources", details: nil))
                }
                return
            }

            let success = KataGoOnnxBridge.initialize(
                withConfig: config,
                modelBin: modelBin,
                modelOnnx: modelOnnx,
                boardSize: Int32(boardSize)
            )

            isOnnxInitialized = success
            DispatchQueue.main.async {
                result(success)
            }
        }
    }

    private func analyzeOnnx(args: [String: Any]?, result: @escaping FlutterResult) {
        guard isOnnxInitialized else {
            result(FlutterError(code: "NOT_INITIALIZED", message: "ONNX engine not initialized", details: nil))
            return
        }

        guard let args = args else {
            result(FlutterError(code: "INVALID_ARGS", message: "Missing arguments", details: nil))
            return
        }

        let maxVisits = args["maxVisits"] as? Int ?? 500

        // Convert moves from ["B Q16", "W D4"] to [["B", "Q16"], ["W", "D4"]]
        var bridgeArgs = args
        if let moves = args["moves"] as? [String] {
            let converted = moves.compactMap { move -> [String]? in
                let parts = move.split(separator: " ").map(String.init)
                return parts.count == 2 ? parts : nil
            }
            bridgeArgs["moves"] = converted
        }

        startProgressTimer(maxVisits: maxVisits)

        DispatchQueue.global(qos: .userInitiated).async {
            let jsonResult = KataGoOnnxBridge.analyzePosition(bridgeArgs)

            DispatchQueue.main.async {
                self.stopProgressTimer()

                if let sink = self.eventSink {
                    let event: [String: Any] = [
                        "type": "onnx_progress",
                        "currentVisits": maxVisits,
                        "maxVisits": maxVisits,
                        "isComplete": true
                    ]
                    sink(event)
                }

                if let jsonResult = jsonResult {
                    result(jsonResult)
                } else {
                    result(FlutterError(code: "ANALYSIS_FAILED", message: "ONNX analysis failed", details: nil))
                }
            }
        }
    }

    private func cancelAnalysisOnnx(result: @escaping FlutterResult) {
        KataGoOnnxBridge.requestStop()
        stopProgressTimer()
        result(true)
    }

    private func stopEngineOnnx(result: FlutterResult) {
        if !isOnnxInitialized {
            result(true)
            return
        }

        KataGoOnnxBridge.destroy()
        isOnnxInitialized = false
        result(true)
    }

    // MARK: - Progress Timer

    private func startProgressTimer(maxVisits: Int) {
        stopProgressTimer()

        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 0.3, repeating: 0.3)
        timer.setEventHandler { [weak self] in
            let currentVisits = KataGoOnnxBridge.getCurrentVisits()
            let max = KataGoOnnxBridge.getMaxVisits()
            DispatchQueue.main.async {
                guard let sink = self?.eventSink else { return }
                let event: [String: Any] = [
                    "type": "onnx_progress",
                    "currentVisits": currentVisits,
                    "maxVisits": max,
                    "isComplete": false
                ]
                sink(event)
            }
        }
        timer.resume()
        progressTimer = timer
    }

    private func stopProgressTimer() {
        progressTimer?.cancel()
        progressTimer = nil
    }

    // MARK: - Resource Preparation

    private func prepareOnnxResources(boardSize: Int) -> (String?, String?, String?) {
        // Plugin assets use package prefix
        let modelBinKey = FlutterDartProject.lookupKey(
            forAsset: "packages/katago_onnx_mobile/assets/katago/model.bin")
        let modelBinKeyFallback = FlutterDartProject.lookupKey(
            forAsset: "assets/katago/model.bin")

        let modelBinPath = Bundle.main.path(forResource: modelBinKey, ofType: nil)
            ?? Bundle.main.path(forResource: modelBinKeyFallback, ofType: nil)

        guard let modelBin = modelBinPath else {
            NSLog("[KataGoONNX] Failed to find model.bin")
            return (nil, nil, nil)
        }

        // ONNX model
        let modelOnnxKey = FlutterDartProject.lookupKey(
            forAsset: "packages/katago_onnx_mobile/assets/katago/model.onnx")
        let modelOnnxKeyFallback = FlutterDartProject.lookupKey(
            forAsset: "assets/katago/model.onnx")

        let modelOnnxPath = Bundle.main.path(forResource: modelOnnxKey, ofType: nil)
            ?? Bundle.main.path(forResource: modelOnnxKeyFallback, ofType: nil)

        guard let modelOnnx = modelOnnxPath else {
            NSLog("[KataGoONNX] Failed to find model.onnx")
            return (nil, nil, nil)
        }

        NSLog("[KataGoONNX] Using model.onnx")

        // Config file
        let docDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let configURL = docDir.appendingPathComponent("analysis_onnx.cfg")

        let configContent = """
            # KataGo Analysis Config for iOS ONNX
            maxVisits = 500
            numSearchThreads = 1
            reportAnalysisWinratesAs = BLACK
            nnCacheSizePowerOfTwo = 18
            nnMutexPoolSizePowerOfTwo = 14
            numNNServerThreadsPerModel = 1
            nnMaxBatchSize = 1
            logSearchInfo = false
            logToStderr = true
            """

        try? configContent.write(to: configURL, atomically: true, encoding: .utf8)

        return (configURL.path, modelBin, modelOnnx)
    }
}

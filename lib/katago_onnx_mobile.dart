/// KataGo ONNX Mobile - Flutter plugin for KataGo AI inference on mobile devices.
///
/// Provides two inference paths:
/// - [OnnxEngine]: Pure Dart ONNX Runtime inference (Android)
/// - [KataGoEngine]: Native C++ KataGo + ONNX via MethodChannel (Android/iOS)
library katago_onnx_mobile;

export 'src/move_candidate.dart';
export 'src/inference_engine.dart';
export 'src/katago_engine.dart';
export 'src/liberty_calculator.dart';
export 'src/tactical_evaluator.dart';

// OnnxEngine uses conditional import: real impl on mobile, stub on web
export 'src/onnx_engine.dart' if (dart.library.html) 'src/onnx_engine_stub.dart';

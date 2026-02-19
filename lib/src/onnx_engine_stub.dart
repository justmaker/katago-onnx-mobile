/// Stub implementation of OnnxEngine for web platform (no dart:ffi support)
library;

import 'inference_engine.dart';
import 'move_candidate.dart';

/// Stub OnnxEngine that is never available (web platform)
class OnnxEngine implements InferenceEngine {
  @override
  String get engineName => 'ONNX Runtime (unavailable)';

  @override
  bool get isAvailable => false;

  @override
  bool get isRunning => false;

  @override
  Future<bool> start({int boardSize = 19}) async => false;

  @override
  Future<void> stop() async {}

  @override
  Future<EngineAnalysisResult> analyze({
    required int boardSize,
    required List<String> moves,
    required double komi,
    required int maxVisits,
    AnalysisProgressCallback? onProgress,
  }) {
    throw UnsupportedError('ONNX engine not available on web');
  }

  @override
  void cancelAnalysis() {}

  @override
  void dispose() {}
}

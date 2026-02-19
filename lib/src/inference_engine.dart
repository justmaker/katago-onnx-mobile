/// Abstract interface for Go AI inference engines.
///
/// Platform-specific implementations:
/// - Android: ONNX Runtime (onnx_engine.dart)
/// - iOS: Native KataGo + ONNX (katago_engine.dart)
/// - Web: Stub (onnx_engine_stub.dart)
library;

import 'move_candidate.dart';

/// Progress information during analysis
class AnalysisProgress {
  final int currentVisits;
  final int maxVisits;
  final double winrate;
  final double scoreLead;
  final String? bestMove;
  final bool isComplete;

  AnalysisProgress({
    required this.currentVisits,
    required this.maxVisits,
    required this.winrate,
    required this.scoreLead,
    this.bestMove,
    this.isComplete = false,
  });

  double get progress => maxVisits > 0 ? currentVisits / maxVisits : 0;
}

/// Analysis progress callback
typedef AnalysisProgressCallback = void Function(AnalysisProgress progress);

/// Analysis result from AI engine
class EngineAnalysisResult {
  final List<MoveCandidate> topMoves;
  final int visits;
  final String modelName;

  EngineAnalysisResult({
    required this.topMoves,
    required this.visits,
    required this.modelName,
  });
}

/// Abstract AI inference engine interface
abstract class InferenceEngine {
  /// Engine name (e.g., "KataGo Native", "ONNX Runtime")
  String get engineName;

  /// Check if engine is available on current platform
  bool get isAvailable;

  /// Check if engine is currently running
  bool get isRunning;

  /// Start the inference engine for the given board size
  Future<bool> start({int boardSize = 19});

  /// Stop the inference engine
  Future<void> stop();

  /// Analyze a board position
  Future<EngineAnalysisResult> analyze({
    required int boardSize,
    required List<String> moves,
    required double komi,
    required int maxVisits,
    AnalysisProgressCallback? onProgress,
  });

  /// Cancel ongoing analysis
  void cancelAnalysis();

  /// Dispose resources
  void dispose();
}

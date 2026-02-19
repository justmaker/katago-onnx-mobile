/// Native KataGo inference engine wrapper via MethodChannel.
///
/// For Android and iOS platforms:
/// - Android: JNI → C++ KataGo + ONNX Runtime
/// - iOS: ObjC++ → C++ KataGo + ONNX Runtime
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'inference_engine.dart';
import 'move_candidate.dart';

/// Native KataGo ONNX engine (mobile platforms)
class KataGoEngine implements InferenceEngine {
  static const String _tag = '[KataGoEngine]';

  /// MethodChannel name. Can be overridden for backward compatibility.
  final String channelName;

  /// EventChannel name for progress updates.
  final String eventChannelName;

  late final MethodChannel _methodChannel;
  late final EventChannel _eventChannel;
  StreamSubscription? _progressSubscription;
  bool _nativeRunning = false;

  KataGoEngine({
    this.channelName = 'com.justmaker.katago_onnx_mobile/engine',
    this.eventChannelName = 'com.justmaker.katago_onnx_mobile/events',
  }) {
    _methodChannel = MethodChannel(channelName);
    _eventChannel = EventChannel(eventChannelName);
  }

  @override
  String get engineName => 'KataGo Mobile (Native ONNX)';

  @override
  bool get isAvailable => !kIsWeb && (Platform.isAndroid || Platform.isIOS);

  @override
  bool get isRunning => _nativeRunning;

  @override
  Future<bool> start({int boardSize = 19}) async {
    if (!isAvailable) {
      debugPrint('$_tag KataGo not available on this platform');
      return false;
    }

    try {
      final methodName = Platform.isIOS ? 'startEngineOnnx' : 'startEngine';
      debugPrint('$_tag Calling $methodName with boardSize=$boardSize');

      final success = await _methodChannel
              .invokeMethod<bool>(methodName, {'boardSize': boardSize}) ??
          false;
      _nativeRunning = success;

      if (success) {
        debugPrint('$_tag Native KataGo ($methodName) started for ${boardSize}x$boardSize');
      }
      return success;
    } catch (e) {
      debugPrint('$_tag Failed to start: $e');
      return false;
    }
  }

  @override
  Future<void> stop() async {
    _nativeRunning = false;
    try {
      await _methodChannel.invokeMethod('stopEngine');
    } catch (e) {
      debugPrint('$_tag Failed to stop: $e');
    }
  }

  @override
  Future<EngineAnalysisResult> analyze({
    required int boardSize,
    required List<String> moves,
    required double komi,
    required int maxVisits,
    AnalysisProgressCallback? onProgress,
  }) async {
    try {
      debugPrint('$_tag analyze() called: ${moves.length} moves, $maxVisits visits');

      final methodName = Platform.isIOS ? 'analyzeOnnx' : 'analyze';

      // On iOS, subscribe to EventChannel for progress updates
      if (Platform.isIOS && onProgress != null) {
        _progressSubscription?.cancel();
        _progressSubscription = _eventChannel.receiveBroadcastStream().listen((event) {
          if (event is Map) {
            final type = event['type'];
            if (type == 'onnx_progress') {
              final currentVisits = (event['currentVisits'] as num?)?.toInt() ?? 0;
              final max = (event['maxVisits'] as num?)?.toInt() ?? maxVisits;
              final isComplete = event['isComplete'] as bool? ?? false;
              onProgress(AnalysisProgress(
                currentVisits: currentVisits,
                maxVisits: max,
                winrate: 0.5,
                scoreLead: 0.0,
                isComplete: isComplete,
              ));
            }
          }
        });
      }

      try {
        final response = await _methodChannel.invokeMethod<String>(methodName, {
          'boardXSize': boardSize,
          'boardYSize': boardSize,
          'moves': moves,
          'komi': komi,
          'maxVisits': maxVisits,
        });

        if (response == null || response.isEmpty) {
          throw Exception('Empty response from native engine');
        }

        debugPrint('$_tag Got response: ${response.length} bytes');
        return _parseNativeResponse(response, maxVisits);
      } finally {
        _progressSubscription?.cancel();
        _progressSubscription = null;
      }
    } catch (e) {
      debugPrint('$_tag analyze() error: $e');
      rethrow;
    }
  }

  /// Parse the JSON response from the native KataGo engine
  EngineAnalysisResult _parseNativeResponse(String jsonStr, int maxVisits) {
    final data = jsonDecode(jsonStr);

    if (data is Map<String, dynamic> && data.containsKey('error')) {
      throw Exception(data['error']);
    }

    final moveInfos = data['moveInfos'] as List? ?? [];
    final rootInfo = data['rootInfo'] as Map<String, dynamic>? ?? {};

    final topMoves = moveInfos.take(10).map((info) {
      final moveInfo = info as Map<String, dynamic>;
      return MoveCandidate(
        move: moveInfo['move'] as String? ?? 'pass',
        winrate: ((moveInfo['winrate'] as num?)?.toDouble() ?? 0.5).clamp(0.0, 1.0),
        scoreLead: (moveInfo['scoreLead'] as num?)?.toDouble() ?? 0.0,
        visits: moveInfo['visits'] as int? ?? 0,
      );
    }).toList();

    final visits = rootInfo['visits'] as int? ?? maxVisits;
    debugPrint('$_tag Parsed ${topMoves.length} moves, $visits visits');

    return EngineAnalysisResult(
      topMoves: topMoves,
      visits: visits,
      modelName: 'katago-native-onnx',
    );
  }

  @override
  void cancelAnalysis() {
    _progressSubscription?.cancel();
    _progressSubscription = null;
    if (!kIsWeb && Platform.isIOS) {
      _methodChannel.invokeMethod('cancelAnalysisOnnx');
    }
  }

  @override
  void dispose() {
    stop();
  }
}

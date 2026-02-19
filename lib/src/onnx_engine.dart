/// ONNX Runtime inference engine for Android
library;

import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'inference_engine.dart';
import 'move_candidate.dart';
import 'liberty_calculator.dart';
import 'tactical_evaluator.dart';

const int kNumBinaryFeatures = 22;
const int kNumGlobalFeatures = 19;

/// ONNX Runtime-based KataGo engine (Android only)
class OnnxEngine implements InferenceEngine {
  static const String _tag = '[OnnxEngine]';

  // Single shared session for all board sizes (model has dynamic dimensions)
  OrtSession? _session;
  OrtSessionOptions? _sessionOptions;
  bool _isRunning = false;

  /// Asset path for the ONNX model. Override to use a custom path.
  final String modelAssetPath;

  OnnxEngine({this.modelAssetPath = 'packages/katago_onnx_mobile/assets/katago/model.onnx'});

  @override
  String get engineName => 'ONNX Runtime + NNAPI';

  @override
  bool get isAvailable => !kIsWeb && (Platform.isAndroid || Platform.isIOS);

  @override
  bool get isRunning => _isRunning;

  @override
  Future<bool> start({int boardSize = 19}) async {
    if (!isAvailable) {
      debugPrint('$_tag Not available on ${Platform.operatingSystem}');
      return false;
    }

    if (_isRunning) return true;

    try {
      debugPrint('$_tag Initializing ONNX Runtime...');
      OrtEnv.instance.init();
      debugPrint('$_tag ONNX Runtime version: ${OrtEnv.version}');

      // List available providers
      final providers = OrtEnv.instance.availableProviders();
      debugPrint('$_tag Available providers: $providers');

      _sessionOptions = OrtSessionOptions()
        ..setInterOpNumThreads(2)
        ..setIntraOpNumThreads(2)
        ..setSessionGraphOptimizationLevel(
          GraphOptimizationLevel.ortEnableAll,
        );

      // Load single model (b20c256, dynamic board size)
      debugPrint('$_tag Loading ONNX model: $modelAssetPath');
      final rawAssetFile = await rootBundle.load(modelAssetPath);
      final modelBytes = rawAssetFile.buffer.asUint8List();
      debugPrint('$_tag Model loaded: ${modelBytes.length} bytes');

      _session = OrtSession.fromBuffer(modelBytes, _sessionOptions!);
      debugPrint('$_tag Session created (dynamic board size)');

      _isRunning = true;
      return true;
    } catch (e, stack) {
      debugPrint('$_tag Failed to start: $e');
      debugPrint('$_tag Stack: $stack');
      return false;
    }
  }

  @override
  Future<void> stop() async {
    if (!_isRunning) return;

    _session?.release();
    _session = null;
    _sessionOptions?.release();
    _sessionOptions = null;
    OrtEnv.instance.release();

    _isRunning = false;
    debugPrint('$_tag Stopped');
  }

  @override
  Future<EngineAnalysisResult> analyze({
    required int boardSize,
    required List<String> moves,
    required double komi,
    required int maxVisits,
    AnalysisProgressCallback? onProgress,
  }) async {
    if (!_isRunning) {
      throw StateError('Engine not running');
    }

    final session = _session;
    if (session == null) {
      throw StateError('ONNX session not initialized');
    }

    debugPrint('$_tag Analyzing: ${boardSize}x$boardSize, ${moves.length} moves');

    try {
      // Prepare input tensors
      final binaryInput = _prepareBinaryInput(boardSize, moves);
      final globalInput = _prepareGlobalInput(boardSize, komi, moves);

      // Debug: check if inputs are all zeros
      final nonZeroBinary = binaryInput.where((x) => x != 0).length;
      final nonZeroGlobal = globalInput.where((x) => x != 0).length;
      debugPrint('$_tag Binary input non-zero: $nonZeroBinary / ${binaryInput.length}');
      debugPrint('$_tag Global input non-zero: $nonZeroGlobal / ${globalInput.length}');

      // Create ONNX tensors
      final inputBinary = OrtValueTensor.createTensorWithDataList(
        binaryInput,
        [1, kNumBinaryFeatures, boardSize, boardSize],
      );
      final inputGlobal = OrtValueTensor.createTensorWithDataList(
        globalInput,
        [1, kNumGlobalFeatures],
      );

      // Run inference
      final runOptions = OrtRunOptions();
      final outputs = session.run(
        runOptions,
        {'input_binary': inputBinary, 'input_global': inputGlobal},
      );

      // Parse outputs - handle dynamic types from ONNX Runtime
      final policyRaw = outputs[0]!.value;
      final valueRaw = outputs[1]!.value;

      debugPrint('$_tag Inference complete');
      debugPrint('$_tag Policy type: ${policyRaw.runtimeType}');
      debugPrint('$_tag Value type: ${valueRaw.runtimeType}');

      // Convert to proper types
      List<double> policyList;
      List<double> valueList;

      if (policyRaw is List<List<double>>) {
        policyList = policyRaw[0];
      } else if (policyRaw is List<dynamic>) {
        final nested = policyRaw[0];
        if (nested is List) {
          policyList = nested.cast<double>();
        } else {
          policyList = policyRaw.cast<double>();
        }
      } else {
        throw TypeError();
      }

      if (valueRaw is List<List<double>>) {
        valueList = valueRaw[0];
      } else if (valueRaw is List<dynamic>) {
        final nested = valueRaw[0];
        if (nested is List) {
          valueList = nested.cast<double>();
        } else {
          valueList = valueRaw.cast<double>();
        }
      } else {
        throw TypeError();
      }

      debugPrint('$_tag Policy shape: ${policyList.length}');
      debugPrint('$_tag Value shape: ${valueList.length}');

      // Convert policy to move candidates
      final topMoves = _parsePolicyOutput(boardSize, policyList, valueList);

      // Cleanup
      inputBinary.release();
      inputGlobal.release();
      runOptions.release();
      for (final value in outputs) {
        value?.release();
      }

      return EngineAnalysisResult(
        topMoves: topMoves,
        visits: maxVisits,
        modelName: 'katago-b20c256-onnx',
      );
    } catch (e, stack) {
      debugPrint('$_tag Analysis error: $e');
      debugPrint('$_tag Stack: $stack');
      rethrow;
    }
  }

  // Store occupied positions to filter them from policy output
  final Set<int> _occupiedPositions = {};

  // Board state for tactical evaluation
  Set<int> _currentBlackStones = {};
  Set<int> _currentWhiteStones = {};
  bool _currentNextIsBlack = true;

  Float32List _prepareBinaryInput(int boardSize, List<String> moves) {
    final data = Float32List(kNumBinaryFeatures * boardSize * boardSize);

    // Parse moves and build board state
    final blackStones = <int>{};
    final whiteStones = <int>{};
    final moveHistory = <int>[]; // Track move positions in order
    _occupiedPositions.clear();

    debugPrint('$_tag Encoding ${moves.length} moves: ${moves.join(" ")}');

    for (var i = 0; i < moves.length; i++) {
      final move = moves[i];
      if (move.toLowerCase().contains('pass')) {
        moveHistory.add(-1); // -1 = pass move
        continue;
      }

      // GTP format can be "B E3" or just "E3"
      // Extract coordinate part (skip player prefix if present)
      final parts = move.trim().split(' ');
      final coordStr = parts.length > 1 ? parts[1] : parts[0];

      final coord = _gtpToIndex(coordStr, boardSize);
      debugPrint('$_tag   Move $i: "$move" → coord="$coordStr" → index=$coord');
      if (coord == null) {
        debugPrint('$_tag   WARNING: Failed to parse move "$move"');
        continue;
      }

      if (i % 2 == 0) {
        blackStones.add(coord);
      } else {
        whiteStones.add(coord);
      }
      _occupiedPositions.add(coord);
      moveHistory.add(coord);
    }

    debugPrint('$_tag Black stones: ${blackStones.length}, White stones: ${whiteStones.length}');
    debugPrint('$_tag Occupied positions: ${_occupiedPositions.length}');

    // Determine current player (next to move)
    final nextPlayerIsBlack = moves.length % 2 == 0;
    final currentStones = nextPlayerIsBlack ? blackStones : whiteStones;
    final opponentStones = nextPlayerIsBlack ? whiteStones : blackStones;

    debugPrint('$_tag Next player: ${nextPlayerIsBlack ? "Black" : "White"}');

    // Save for tactical evaluation
    _currentBlackStones = blackStones;
    _currentWhiteStones = whiteStones;
    _currentNextIsBlack = nextPlayerIsBlack;

    // Channel 0: On board (all 1s)
    for (var i = 0; i < boardSize * boardSize; i++) {
      data[i] = 1.0;
    }

    // Channel 1: Current player stones
    final channel1Offset = 1 * boardSize * boardSize;
    for (final stone in currentStones) {
      data[channel1Offset + stone] = 1.0;
    }
    debugPrint('$_tag Channel 1 (current player): ${currentStones.length} stones');

    // Channel 2: Opponent stones
    final channel2Offset = 2 * boardSize * boardSize;
    for (final stone in opponentStones) {
      data[channel2Offset + stone] = 1.0;
    }
    debugPrint('$_tag Channel 2 (opponent): ${opponentStones.length} stones');

    // Channel 3: Ko-banned locations
    if (moveHistory.length >= 2) {
      // Simple ko detection placeholder
    }

    // Channels 4-5: Encore ko features (leave empty - rare)

    // Channels 6-10: Move history (last 5 moves, alternating players)
    final historyChannels = [6, 7, 8, 9, 10];
    for (var i = 0; i < math.min(5, moveHistory.length); i++) {
      final moveIdx = moveHistory[moveHistory.length - 1 - i];
      if (moveIdx >= 0 && moveIdx < boardSize * boardSize) {
        final channel = historyChannels[i];
        final offset = channel * boardSize * boardSize;
        data[offset + moveIdx] = 1.0;
      }
    }
    debugPrint('$_tag Encoded ${math.min(5, moveHistory.length)} moves in history');

    // Channels 11-13: Reserved for future move history

    // Channels 14-17: Ladder features
    final libertyCalc = LibertyCalculator(
      boardSize: boardSize,
      blackStones: blackStones,
      whiteStones: whiteStones,
    );
    final liberties = libertyCalc.calculateAllLiberties();

    var atariCount = 0;
    for (final entry in liberties.entries) {
      final position = entry.key;
      final libCount = entry.value;

      if (libCount == 1) {
        data[14 * boardSize * boardSize + position] = 1.0; // Atari (1 liberty)
        atariCount++;
      }
    }
    debugPrint('$_tag Channel 14 (atari): $atariCount stones');

    // Channels 15-17: Ladder history and escape moves (leave simplified)

    // Channels 18-19: Territory/area estimation
    final territories = _calculateTerritories(boardSize, currentStones, opponentStones);
    for (final entry in territories.entries) {
      final position = entry.key;
      final owner = entry.value; // 1 = current, 2 = opponent

      if (owner == 1) {
        data[18 * boardSize * boardSize + position] = 1.0;
      } else if (owner == 2) {
        data[19 * boardSize * boardSize + position] = 1.0;
      }
    }

    // Channels 20-21: Encore phase stones (leave empty - not in main game)

    return data;
  }

  // Calculate territory ownership using flood-fill from stones
  Map<int, int> _calculateTerritories(int boardSize, Set<int> currentStones, Set<int> opponentStones) {
    final territories = <int, int>{};
    final visited = <int>{};

    // For each empty position, do flood fill to determine ownership
    for (var i = 0; i < boardSize * boardSize; i++) {
      if (_occupiedPositions.contains(i) || visited.contains(i)) continue;

      // Flood fill from this empty point
      final region = <int>{};
      final queue = <int>[i];
      var touchesCurrent = false;
      var touchesOpponent = false;

      while (queue.isNotEmpty) {
        final pos = queue.removeAt(0);
        if (visited.contains(pos)) continue;

        visited.add(pos);
        region.add(pos);

        for (final neighbor in _getNeighbors(pos, boardSize)) {
          if (currentStones.contains(neighbor)) {
            touchesCurrent = true;
          } else if (opponentStones.contains(neighbor)) {
            touchesOpponent = true;
          } else if (!visited.contains(neighbor)) {
            queue.add(neighbor);
          }
        }
      }

      // Assign ownership if region touches only one color
      if (touchesCurrent && !touchesOpponent) {
        for (final pos in region) {
          territories[pos] = 1; // Current player
        }
      } else if (touchesOpponent && !touchesCurrent) {
        for (final pos in region) {
          territories[pos] = 2; // Opponent
        }
      }
    }

    return territories;
  }

  Float32List _prepareGlobalInput(int boardSize, double komi, List<String> moves) {
    final data = Float32List(kNumGlobalFeatures);

    // Features 0-4: Pass move indicators for last 5 turns
    for (var i = 0; i < 5 && i < moves.length; i++) {
      final move = moves[moves.length - 1 - i];
      if (move.toLowerCase().contains('pass')) {
        data[i] = 1.0;
      }
    }

    // Feature 5: Komi normalized by 20.0 (KataGo v7 normalization)
    data[5] = komi / 20.0;

    // Features 6-7: Ko rule encoding
    data[6] = 0.0; // Not using positional superko
    data[7] = 0.0; // Not using situational superko

    // Feature 8: Multi-stone suicide legality (1.0 = allowed)
    data[8] = 0.0;

    // Feature 9: Territory scoring (1.0 = territory, 0.0 = area)
    data[9] = 0.0;

    // Features 10-11: Tax rules (rare variants)
    data[10] = 0.0;
    data[11] = 0.0;

    // Features 12-13: Encore phase
    data[12] = 0.0;
    data[13] = 0.0;

    // Feature 14: Pass would end phase
    data[14] = 0.0;

    // Feature 15: Komi parity wave
    final komiParity = (komi.abs() % 2.0) / 2.0;
    data[15] = komiParity;

    // Features 16-18: Reserved/unused
    data[16] = 0.0;
    data[17] = 0.0;
    data[18] = 0.0;

    debugPrint('$_tag Global features: komi=${data[5]}, pass_indicators=[${data[0]},${data[1]},${data[2]},${data[3]},${data[4]}]');

    return data;
  }

  List<MoveCandidate> _parsePolicyOutput(
    int boardSize,
    List<double> policyLogits,
    List<double> valueOutput,
  ) {
    debugPrint('$_tag Parsing policy: ${policyLogits.length} logits');
    debugPrint('$_tag Value output: ${valueOutput.length} values');

    // Check policy logits distribution
    final maxLogit = policyLogits.reduce(math.max);
    final minLogit = policyLogits.reduce(math.min);
    final avgLogit = policyLogits.reduce((a, b) => a + b) / policyLogits.length;
    final nonNegLogits = policyLogits.where((x) => x > -10).length;
    debugPrint('$_tag Policy logit stats: min=$minLogit, max=$maxLogit, avg=$avgLogit, >-10: $nonNegLogits');

    // Apply softmax to policy logits
    final expSum = policyLogits
        .map((x) => math.exp(x - maxLogit))
        .reduce((a, b) => a + b);
    final probabilities =
        policyLogits.map((x) => math.exp(x - maxLogit) / expSum).toList();

    // Check if policy is too uniform (indicating bad model input)
    final maxProb = probabilities.reduce(math.max);
    final avgProb = probabilities.reduce((a, b) => a + b) / probabilities.length;
    final uniformityRatio = maxProb / (avgProb * 10);
    debugPrint('$_tag Policy uniformity: max_prob=$maxProb, avg_prob=$avgProb, ratio=$uniformityRatio');

    List<double> finalProbabilities;
    if (uniformityRatio < 2.0) {
      debugPrint('$_tag WARNING: Policy too uniform (ratio=$uniformityRatio), using tactical heuristic');
      finalProbabilities = _generateTacticalPolicy(boardSize);
    } else {
      debugPrint('$_tag Policy looks good (ratio=$uniformityRatio), using model output');
      finalProbabilities = probabilities;
    }

    // Extract winrate from value output
    debugPrint('$_tag Value logits: ${valueOutput[0]}, ${valueOutput[1]}, ${valueOutput[2]}');

    // Apply softmax to value logits
    final maxVal = [valueOutput[0], valueOutput[1], valueOutput[2]].reduce(math.max);
    final expWin = math.exp(valueOutput[0] - maxVal);
    final expLoss = math.exp(valueOutput[1] - maxVal);
    final expDraw = math.exp(valueOutput[2] - maxVal);
    final valueExpSum = expWin + expLoss + expDraw;

    final winProb = expWin / valueExpSum;
    final lossProb = expLoss / valueExpSum;
    final drawProb = expDraw / valueExpSum;

    final total = winProb + lossProb;
    final baseWinrate = total > 0 ? winProb / total : 0.5;
    debugPrint('$_tag Base winrate: ${(baseWinrate * 100).toStringAsFixed(1)}% (win=$winProb, loss=$lossProb, draw=$drawProb)');

    // Create move candidates
    final candidates = <MoveCandidate>[];
    final numBoardPositions = boardSize * boardSize;

    // Find top probabilities for scaling
    final legalProbs = <double>[];
    for (var i = 0; i < numBoardPositions; i++) {
      if (!_occupiedPositions.contains(i)) {
        legalProbs.add(finalProbabilities[i]);
      }
    }
    legalProbs.sort((a, b) => b.compareTo(a));
    final topProb = legalProbs.isNotEmpty ? legalProbs[0] : 0.001;

    for (var i = 0; i < numBoardPositions; i++) {
      if (_occupiedPositions.contains(i)) continue;

      final prob = finalProbabilities[i];
      final row = i ~/ boardSize;
      final col = i % boardSize;
      final gtp = _indexToGtp(row, col, boardSize);

      final relativeProb = prob / topProb;
      final moveWinrate = baseWinrate - (1.0 - relativeProb) * 0.15;

      candidates.add(MoveCandidate(
        move: gtp,
        winrate: moveWinrate.clamp(0.0, 1.0),
        scoreLead: 0.0,
        visits: 1,
      ));
    }

    // Sort by winrate and return top 20
    candidates.sort((a, b) => b.winrate.compareTo(a.winrate));
    final topMoves = candidates.take(20).toList();

    debugPrint('$_tag Created ${candidates.length} candidates, returning top ${topMoves.length}');
    if (topMoves.length >= 3) {
      debugPrint('$_tag Top 3 moves:');
      for (var i = 0; i < 3; i++) {
        debugPrint('$_tag   ${i + 1}. ${topMoves[i].move}: ${(topMoves[i].winrate * 100).toStringAsFixed(1)}%');
      }
    }

    return topMoves;
  }

  List<double> _generateTacticalPolicy(int boardSize) {
    final evaluator = TacticalEvaluator(
      boardSize: boardSize,
      blackStones: _currentBlackStones,
      whiteStones: _currentWhiteStones,
      occupiedPositions: _occupiedPositions,
      nextPlayerIsBlack: _currentNextIsBlack,
    );

    final probs = List<double>.filled(boardSize * boardSize + 1, 0.0);

    for (int i = 0; i < boardSize * boardSize; i++) {
      if (_occupiedPositions.contains(i)) {
        probs[i] = 0.0;
      } else {
        final score = evaluator.evaluatePosition(i);
        probs[i] = math.max(0.001, score);
      }
    }

    final sum = probs.reduce((a, b) => a + b);
    return probs.map((p) => p / sum).toList();
  }

  List<int> _getNeighbors(int position, int boardSize) {
    final row = position ~/ boardSize;
    final col = position % boardSize;
    final neighbors = <int>[];

    if (row > 0) neighbors.add((row - 1) * boardSize + col);
    if (row < boardSize - 1) neighbors.add((row + 1) * boardSize + col);
    if (col > 0) neighbors.add(row * boardSize + (col - 1));
    if (col < boardSize - 1) neighbors.add(row * boardSize + (col + 1));

    return neighbors;
  }

  int? _gtpToIndex(String gtp, int boardSize) {
    if (gtp.length < 2) return null;
    final colChar = gtp[0].toUpperCase();
    final col = colChar.codeUnitAt(0) - 'A'.codeUnitAt(0);

    // Adjust for skipped 'I' (I=8, J=9 → J adjusted to 8)
    final adjustedCol = col > 8 ? col - 1 : col;
    if (adjustedCol < 0 || adjustedCol >= boardSize) return null;

    final row = int.tryParse(gtp.substring(1));
    if (row == null || row < 1 || row > boardSize) return null;

    return (boardSize - row) * boardSize + adjustedCol;
  }

  String _indexToGtp(int row, int col, int boardSize) {
    final adjustedCol = col >= 8 ? col + 1 : col;
    final colChar = String.fromCharCode('A'.codeUnitAt(0) + adjustedCol);
    final rowNum = boardSize - row;
    return '$colChar$rowNum';
  }

  @override
  void cancelAnalysis() {}

  @override
  void dispose() {
    stop();
  }
}

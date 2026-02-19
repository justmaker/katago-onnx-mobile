/// Tactical move evaluator for Go
/// Evaluates moves based on current board situation
library;

import 'dart:math' as math;
import 'liberty_calculator.dart';

class TacticalEvaluator {
  final int boardSize;
  final Set<int> blackStones;
  final Set<int> whiteStones;
  final Set<int> occupiedPositions;
  final bool nextPlayerIsBlack;

  TacticalEvaluator({
    required this.boardSize,
    required this.blackStones,
    required this.whiteStones,
    required this.occupiedPositions,
    required this.nextPlayerIsBlack,
  });

  /// Evaluate a position tactically
  /// Returns score (higher = better)
  double evaluatePosition(int position) {
    var score = 0.0;

    // 1. MOST CRITICAL: Save our stones from atari
    final saveScore = _evaluateSave(position);
    if (saveScore > 0) {
      score += saveScore * 200; // Saving is MOST important
    }

    // 2. Capture opponent stones (but less important than saving)
    final captureScore = _evaluateCapture(position);
    if (captureScore > 0) {
      score += captureScore * 80; // Capturing is important
    }

    // 3. Check if this move attacks opponent (puts them in atari)
    final attackScore = _evaluateAttack(position);
    if (attackScore > 0) {
      score += attackScore * 30; // Attacking is good
    }

    // 4. Check if this move extends our territory
    final territoryScore = _evaluateTerritory(position);
    score += territoryScore * 10;

    // 5. Position value (opening principles)
    final positionScore = _evaluatePositionValue(position);
    score += positionScore;

    return score;
  }

  /// Check if playing here captures opponent stones
  double _evaluateCapture(int position) {
    final myStones = nextPlayerIsBlack ? blackStones : whiteStones;
    final oppStones = nextPlayerIsBlack ? whiteStones : blackStones;

    // Simulate placing stone
    final testMyStones = Set<int>.from(myStones)..add(position);
    final testOppStones = Set<int>.from(oppStones);

    // Check all opponent neighbors
    final neighbors = _getNeighbors(position);
    var captureValue = 0.0;

    for (final n in neighbors) {
      if (oppStones.contains(n)) {
        // Check if this opponent group would have 0 liberties
        final libs = _calculateLibertiesAfterMove(n, testOppStones, testMyStones);
        if (libs == 0) {
          // This move captures! Count group size
          final groupSize = _getGroupSize(n, oppStones);
          captureValue += groupSize; // Bigger captures are better

          // BUT: Check if we get captured back (2-ply reading)
          final testOccupied = Set<int>.from(occupiedPositions)..add(position);
          testOccupied.removeAll(_getGroup(n, oppStones)); // Remove captured stones

          // Simulate opponent's response at our new stone
          for (final counterMove in _getNeighbors(position)) {
            if (!testOccupied.contains(counterMove)) {
              final testCounterOpp = Set<int>.from(testOppStones)..add(counterMove);
              final ourLibsAfter = _calculateLibertiesAfterMove(position, testMyStones, testCounterOpp);
              if (ourLibsAfter == 0) {
                // We get captured back - bad exchange!
                captureValue *= 0.3; // 70% penalty
                break;
              }
            }
          }
        }
      }
    }

    return captureValue;
  }

  /// Check if playing here saves our stones from atari
  double _evaluateSave(int position) {
    final myStones = nextPlayerIsBlack ? blackStones : whiteStones;

    // Check if any of our stones are in atari (1 liberty)
    final libertyCalc = LibertyCalculator(
      boardSize: boardSize,
      blackStones: blackStones,
      whiteStones: whiteStones,
    );
    final liberties = libertyCalc.calculateAllLiberties();

    // Check if playing here saves any atari groups
    final neighbors = _getNeighbors(position);
    for (final n in neighbors) {
      if (myStones.contains(n)) {
        final libs = liberties[n] ?? 0;
        if (libs == 1) {
          // Our stone is in atari, playing here might save it
          return 1.0;
        }
      }
    }

    return 0.0;
  }

  /// Check if playing here puts opponent in atari
  double _evaluateAttack(int position) {
    final myStones = nextPlayerIsBlack ? blackStones : whiteStones;
    final oppStones = nextPlayerIsBlack ? whiteStones : blackStones;

    final testMyStones = Set<int>.from(myStones)..add(position);

    final neighbors = _getNeighbors(position);
    for (final n in neighbors) {
      if (oppStones.contains(n)) {
        final libs = _calculateLibertiesAfterMove(n, oppStones, testMyStones);
        if (libs == 1) {
          // This puts opponent in atari
          return 1.0;
        }
      }
    }

    return 0.0;
  }

  /// Evaluate territorial value
  double _evaluateTerritory(int position) {
    // Empty position near our stones = potential territory
    final myStones = nextPlayerIsBlack ? blackStones : whiteStones;
    final oppStones = nextPlayerIsBlack ? whiteStones : blackStones;

    final neighbors = _getNeighbors(position);
    var friendlyNeighbors = 0;
    var enemyNeighbors = 0;

    for (final n in neighbors) {
      if (myStones.contains(n)) friendlyNeighbors++;
      if (oppStones.contains(n)) enemyNeighbors++;
    }

    if (friendlyNeighbors > enemyNeighbors) {
      return friendlyNeighbors.toDouble();
    }
    return 0.0;
  }

  /// Evaluate position value (opening principles)
  double _evaluatePositionValue(int position) {
    final row = position ~/ boardSize;
    final col = position % boardSize;

    final minDistToEdge = math.min(
      math.min(row, boardSize - 1 - row),
      math.min(col, boardSize - 1 - col)
    );

    // Line-based scoring
    if (minDistToEdge == 0) return 0.1; // Edge
    if (minDistToEdge == 1) return 0.5; // 2nd line
    if (minDistToEdge == 2) return 2.0; // 3rd line (excellent)
    if (minDistToEdge == 3) return 2.5; // 4th line (best)
    if (minDistToEdge == 4) return 1.5; // 5th line
    return 1.0; // Center

  }

  List<int> _getNeighbors(int position) {
    final row = position ~/ boardSize;
    final col = position % boardSize;
    final neighbors = <int>[];

    if (row > 0) neighbors.add((row - 1) * boardSize + col);
    if (row < boardSize - 1) neighbors.add((row + 1) * boardSize + col);
    if (col > 0) neighbors.add(row * boardSize + (col - 1));
    if (col < boardSize - 1) neighbors.add(row * boardSize + (col + 1));

    return neighbors;
  }

  int _calculateLibertiesAfterMove(int stone, Set<int> groupStones, Set<int> opponentStones) {
    // Simplified liberty calculation for one group
    final libertyCalc = LibertyCalculator(
      boardSize: boardSize,
      blackStones: nextPlayerIsBlack ? opponentStones : groupStones,
      whiteStones: nextPlayerIsBlack ? groupStones : opponentStones,
    );
    return libertyCalc.calculateLiberties(stone);
  }

  Set<int> _getGroup(int stone, Set<int> groupStones) {
    final group = <int>{};
    final queue = [stone];
    group.add(stone);
    while (queue.isNotEmpty) {
      final pos = queue.removeLast();
      for (final n in _getNeighbors(pos)) {
        if (groupStones.contains(n) && !group.contains(n)) {
          group.add(n);
          queue.add(n);
        }
      }
    }
    return group;
  }

  int _getGroupSize(int stone, Set<int> groupStones) {
    return _getGroup(stone, groupStones).length;
  }
}

/// Liberty (氣數) calculation for Go stones
library;

import 'dart:collection';

class LibertyCalculator {
  final int boardSize;
  final Set<int> blackStones;
  final Set<int> whiteStones;

  LibertyCalculator({
    required this.boardSize,
    required this.blackStones,
    required this.whiteStones,
  });

  /// Calculate liberties for a stone at given position
  /// Returns number of liberties for the group containing this stone
  int calculateLiberties(int position) {
    // Already visited stones in this group
    final visited = <int>{};
    // Liberties (empty adjacent positions) found
    final liberties = <int>{};

    // Determine stone color
    final isBlack = blackStones.contains(position);
    final isWhite = whiteStones.contains(position);
    if (!isBlack && !isWhite) return 0;

    final groupStones = isBlack ? blackStones : whiteStones;

    // BFS to find all stones in this group and their liberties
    final queue = Queue<int>();
    queue.add(position);
    visited.add(position);

    while (queue.isNotEmpty) {
      final pos = queue.removeFirst();
      final neighbors = _getNeighbors(pos);

      for (final neighbor in neighbors) {
        // Empty position = liberty
        if (!blackStones.contains(neighbor) && !whiteStones.contains(neighbor)) {
          liberties.add(neighbor);
        }
        // Same color stone = part of group
        else if (groupStones.contains(neighbor) && !visited.contains(neighbor)) {
          visited.add(neighbor);
          queue.add(neighbor);
        }
      }
    }

    return liberties.length;
  }

  /// Get valid neighbor positions (up, down, left, right)
  List<int> _getNeighbors(int position) {
    final row = position ~/ boardSize;
    final col = position % boardSize;
    final neighbors = <int>[];

    // Up
    if (row > 0) neighbors.add((row - 1) * boardSize + col);
    // Down
    if (row < boardSize - 1) neighbors.add((row + 1) * boardSize + col);
    // Left
    if (col > 0) neighbors.add(row * boardSize + (col - 1));
    // Right
    if (col < boardSize - 1) neighbors.add(row * boardSize + (col + 1));

    return neighbors;
  }

  /// Get all groups and their liberties
  /// Returns Map<position, libertyCount> for all stones
  Map<int, int> calculateAllLiberties() {
    final result = <int, int>{};
    final processed = <int>{};

    for (final stone in [...blackStones, ...whiteStones]) {
      if (!processed.contains(stone)) {
        final libs = calculateLiberties(stone);

        // Mark all stones in this group with same liberty count
        final isBlack = blackStones.contains(stone);
        final groupStones = isBlack ? blackStones : whiteStones;
        final visited = <int>{};
        final queue = Queue<int>();
        queue.add(stone);
        visited.add(stone);

        while (queue.isNotEmpty) {
          final pos = queue.removeFirst();
          result[pos] = libs;
          processed.add(pos);

          for (final neighbor in _getNeighbors(pos)) {
            if (groupStones.contains(neighbor) && !visited.contains(neighbor)) {
              visited.add(neighbor);
              queue.add(neighbor);
            }
          }
        }
      }
    }

    return result;
  }
}

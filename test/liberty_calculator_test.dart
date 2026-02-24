import 'package:flutter_test/flutter_test.dart';
import 'package:katago_onnx_mobile/katago_onnx_mobile.dart';

void main() {
  group('LibertyCalculator', () {
    const int size = 19;

    test('Empty board returns 0 liberties for any position', () {
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {},
        whiteStones: {},
      );

      // Should return 0 because there is no stone at the given position
      expect(calculator.calculateLiberties(0), equals(0));
      expect(calculator.calculateLiberties(size * size ~/ 2), equals(0));
    });

    test('Single stone in center has 4 liberties', () {
      final center = (size ~/ 2) * size + (size ~/ 2);
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {center},
        whiteStones: {},
      );

      expect(calculator.calculateLiberties(center), equals(4));
    });

    test('Single stone on edge has 3 liberties', () {
      const edge = size ~/ 2; // Top edge center
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {edge},
        whiteStones: {},
      );

      expect(calculator.calculateLiberties(edge), equals(3));
    });

    test('Single stone in corner has 2 liberties', () {
      const corner = 0; // Top-left corner
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {corner},
        whiteStones: {},
      );

      expect(calculator.calculateLiberties(corner), equals(2));
    });

    test('Captured stone has 0 liberties', () {
      const pos = size * 2 + 2; // (2, 2)
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {pos},
        whiteStones: {
          pos - size, // Up
          pos + size, // Down
          pos - 1,    // Left
          pos + 1,    // Right
        },
      );

      expect(calculator.calculateLiberties(pos), equals(0));
    });

    test('Small group of stones calculates shared liberties correctly', () {
      const pos1 = size * 2 + 2; // (2, 2)
      const pos2 = size * 2 + 3; // (2, 3)
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {pos1, pos2},
        whiteStones: {},
      );

      // pos1 neighbors: (1, 2), (3, 2), (2, 1), (2, 3) <- (2, 3) is pos2
      // pos2 neighbors: (1, 3), (3, 3), (2, 2), (2, 4) <- (2, 2) is pos1
      // Total unique empty neighbors: (1,2), (3,2), (2,1), (1,3), (3,3), (2,4) -> 6 liberties
      expect(calculator.calculateLiberties(pos1), equals(6));
      expect(calculator.calculateLiberties(pos2), equals(6));
    });

    test('calculateAllLiberties returns correct map', () {
      const blackPos = 0;
      const whitePos = size * size - 1;
      final calculator = LibertyCalculator(
        boardSize: size,
        blackStones: {blackPos},
        whiteStones: {whitePos},
      );

      final result = calculator.calculateAllLiberties();
      expect(result[blackPos], equals(2));
      expect(result[whitePos], equals(2));
      expect(result.length, equals(2));
    });
  });
}

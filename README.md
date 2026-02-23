# KataGo ONNX Mobile

Flutter plugin for KataGo AI inference on mobile devices (Android/iOS).
This plugin provides a way to run KataGo, a state-of-the-art Go AI, directly on mobile devices using ONNX Runtime or native integration.

## Features

- **Two Inference Engines**:
  - `KataGoEngine`: Uses native C++ KataGo implementation via MethodChannel (Android & iOS). Recommended for performance.
  - `OnnxEngine`: Uses pure Dart ONNX Runtime inference (primarily Android).
- **Analysis**: Calculate top moves, winrate, and score lead.
- **Progress Updates**: Get real-time feedback during analysis (visits count).
- **Tactical Evaluation**: Basic tactical features for move candidate generation.

## Installation

Add this to your package's `pubspec.yaml` file:

```yaml
dependencies:
  katago_onnx_mobile:
    git:
      url: https://github.com/justmaker/katago_onnx_mobile.git
```

## Usage

### 1. Import the package

```dart
import 'package:katago_onnx_mobile/katago_onnx_mobile.dart';
```

### 2. Choose and Start an Engine

You can use either `KataGoEngine` (native implementation) or `OnnxEngine` (Dart wrapper).

```dart
// Initialize the engine
InferenceEngine engine = KataGoEngine();
// or use OnnxEngine();

// Check availability
if (!engine.isAvailable) {
  print('Engine not available on this platform');
  return;
}

// Start the engine (load model)
// The boardSize parameter is optional (default 19)
bool success = await engine.start(boardSize: 19);

if (success) {
  print('Engine started successfully');
} else {
  print('Failed to start engine');
}
```

### 3. Analyze a Position

Pass the current board state (moves played) to the `analyze` method.

```dart
try {
  final result = await engine.analyze(
    boardSize: 19,
    moves: [], // List of GTP moves, e.g., ["Q16", "D4", "R16"]
    komi: 6.5,
    maxVisits: 100, // Number of visits for analysis
    onProgress: (progress) {
      print('Progress: ${progress.currentVisits}/${progress.maxVisits}');
    },
  );

  print('Analysis Complete. Visits: ${result.visits}');

  // Print top suggested moves
  for (var move in result.topMoves) {
    print('Move: ${move.move}');
    print('Winrate: ${move.winratePercent}');
    print('Score Lead: ${move.scoreLeadFormatted}');
  }
} catch (e) {
  print('Analysis error: $e');
}
```

### 4. Stop the Engine

Don't forget to stop the engine when you are done to release resources.

```dart
await engine.stop();
```

## Example

See the `example` directory for a complete sample application.

```dart
// Simplified example
import 'package:flutter/material.dart';
import 'package:katago_onnx_mobile/katago_onnx_mobile.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // ... (See example/lib/main.dart for full implementation)
}
```

## API Reference

### InferenceEngine

The base interface for all engines.

- `start({int boardSize})`: Initialize the engine and load the model. Returns `true` if successful.
- `analyze(...)`: Run analysis on a given board position. Returns `EngineAnalysisResult`.
- `stop()`: Release resources.
- `isAvailable`: Check if the engine is supported on the current device.
- `isRunning`: Check if the engine is currently active.

### MoveCandidate

Represents a suggested move by the AI.

- `move`: The move coordinate in GTP format (e.g., "Q16", "pass").
- `winrate`: Win probability (0.0 - 1.0).
- `scoreLead`: Estimated score lead (positive for black, negative for white).
- `visits`: Number of visits allocated to this move.

## Platform Support

| Feature | Android | iOS |
|---|---|---|
| KataGoEngine | ✅ | ✅ |
| OnnxEngine | ✅ | ❓ (Unverified) |

**Note**: `KataGoEngine` is recommended for production use as it leverages native optimizations and platform-specific implementations.

## License

See the LICENSE file for details.

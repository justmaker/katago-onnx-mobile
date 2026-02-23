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
  InferenceEngine? _engine;
  EngineAnalysisResult? _result;
  bool _isAnalyzing = false;
  String _status = 'Idle';

  @override
  void dispose() {
    _engine?.stop();
    super.dispose();
  }

  Future<void> _startEngine(bool useNative) async {
    if (_engine != null) {
      await _engine!.stop();
    }

    setState(() {
      _status = 'Starting ${useNative ? "Native" : "ONNX"} Engine...';
      _result = null;
    });

    try {
      if (useNative) {
        _engine = KataGoEngine();
      } else {
        _engine = OnnxEngine();
      }

      if (!_engine!.isAvailable) {
        setState(() {
          _status = 'Engine not available on this platform';
        });
        return;
      }

      final success = await _engine!.start(boardSize: 19);
      setState(() {
        _status = success ? 'Engine Started (${_engine!.engineName})' : 'Failed to start engine';
      });
    } catch (e) {
      setState(() {
        _status = 'Error starting engine: $e';
      });
    }
  }

  Future<void> _analyze() async {
    if (_engine == null || !_engine!.isRunning) {
      setState(() {
        _status = 'Engine not running';
      });
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _status = 'Analyzing...';
      _result = null;
    });

    try {
      // Example position (empty board)
      final result = await _engine!.analyze(
        boardSize: 19,
        moves: [], // Empty list for starting position
        komi: 6.5,
        maxVisits: 50,
        onProgress: (progress) {
          setState(() {
            _status = 'Analyzing... ${progress.currentVisits}/${progress.maxVisits} visits';
          });
        },
      );

      if (!mounted) return;

      setState(() {
        _result = result;
        _isAnalyzing = false;
        _status = 'Analysis Complete';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isAnalyzing = false;
        _status = 'Analysis Failed: $e';
      });
    }
  }

  Future<void> _stopEngine() async {
    if (_engine != null) {
      await _engine!.stop();
      setState(() {
        _engine = null;
        _status = 'Engine Stopped';
        _result = null;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('KataGo Mobile Example'),
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Status: $_status', style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: _isAnalyzing ? null : () => _startEngine(true),
                    child: const Text('Start Native'),
                  ),
                  ElevatedButton(
                    onPressed: _isAnalyzing ? null : () => _startEngine(false),
                    child: const Text('Start ONNX'),
                  ),
                ],
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: (_engine != null && _engine!.isRunning && !_isAnalyzing) ? _analyze : null,
                child: const Text('Analyze Empty Board'),
              ),
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: _engine != null ? _stopEngine : null,
                style: ElevatedButton.styleFrom(backgroundColor: Colors.redAccent, foregroundColor: Colors.white),
                child: const Text('Stop Engine'),
              ),
              const SizedBox(height: 16),
              if (_result != null) ...[
                Text('Top Moves (${_result!.visits} visits):', style: const TextStyle(fontWeight: FontWeight.bold)),
                Expanded(
                  child: ListView.builder(
                    itemCount: _result!.topMoves.length,
                    itemBuilder: (context, index) {
                      final move = _result!.topMoves[index];
                      return ListTile(
                        title: Text(move.move),
                        subtitle: Text('Winrate: ${move.winratePercent}, Score: ${move.scoreLeadFormatted}'),
                        trailing: Text('${move.visits} visits'),
                      );
                    },
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

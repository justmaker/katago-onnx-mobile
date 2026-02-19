/// Data model for a Go move candidate from AI analysis.
library;

class MoveCandidate {
  final String move;
  final double winrate;
  final double scoreLead;
  final int visits;

  MoveCandidate({
    required this.move,
    required this.winrate,
    required this.scoreLead,
    required this.visits,
  });

  factory MoveCandidate.fromJson(Map<String, dynamic> json) {
    return MoveCandidate(
      move: json['move'] as String,
      winrate: (json['winrate'] as num).toDouble(),
      // Support both snake_case (API) and camelCase (opening book)
      scoreLead: (json['scoreLead'] ?? json['score_lead'] as num).toDouble(),
      visits: json['visits'] as int,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'move': move,
      'winrate': winrate,
      'score_lead': scoreLead,
      'visits': visits,
    };
  }

  /// Winrate as percentage string
  String get winratePercent => '${(winrate * 100).toStringAsFixed(1)}%';

  /// Score lead with sign
  String get scoreLeadFormatted {
    final sign = scoreLead >= 0 ? '+' : '';
    return '$sign${scoreLead.toStringAsFixed(1)}';
  }

  @override
  String toString() {
    return 'MoveCandidate($move, wr=$winratePercent, lead=$scoreLeadFormatted, visits=$visits)';
  }
}

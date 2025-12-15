class Recommendation {
  final String jobId;
  final double finalScore;
  final double contentScore;
  final double lfmScore;

  const Recommendation({
    required this.jobId,
    required this.finalScore,
    required this.contentScore,
    required this.lfmScore,
  });

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    return Recommendation(
      jobId: json['job_id']?.toString() ?? '',
      finalScore: (json['final_score'] ?? 0).toDouble(),
      contentScore: (json['content_score'] ?? 0).toDouble(),
      lfmScore: (json['lfm_score'] ?? 0).toDouble(),
    );
  }
}

import 'job.dart';

class Recommendation {
  final String jobId;
  final String? jobTitle;
  final String? company;
  final String? location;
  final String? category;
  final String? salary;
  final double finalScore;
  final double contentScore;
  final double lfmScore;

  const Recommendation({
    required this.jobId,
    this.jobTitle,
    this.company,
    this.location,
    this.category,
    this.salary,
    required this.finalScore,
    required this.contentScore,
    required this.lfmScore,
  });

  factory Recommendation.fromJson(Map<String, dynamic> json) {
    return Recommendation(
      jobId: json['job_id']?.toString() ?? '',
      jobTitle: json['job_title']?.toString(),
      company: json['company']?.toString(),
      location: json['location']?.toString(),
      category: json['category']?.toString(),
      salary: json['salary']?.toString(),
      finalScore: (json['final_score'] ?? 0).toDouble(),
      contentScore: (json['content_score'] ?? 0).toDouble(),
      lfmScore: (json['lfm_score'] ?? 0).toDouble(),
    );
  }

  /// Convert Recommendation to Job for compatibility
  Job toJob() {
    return Job(
      jobId: jobId,
      jobTitle: jobTitle ?? 'Job',
      company: company,
      location: location,
      category: category,
      salary: salary,
      descriptions: null,
      status: null,
    );
  }
}

class Job {
  final String jobId;
  final String jobTitle;
  final String? company;
  final String? location;
  final String? category;
  final String? salary;
  final String? descriptions;

  const Job({
    required this.jobId,
    required this.jobTitle,
    this.company,
    this.location,
    this.category,
    this.salary,
    this.descriptions,
  });

  String get snippet {
    if ((descriptions ?? '').length < 120) return descriptions ?? '';
    return '${descriptions!.substring(0, 120)}...';
  }

  factory Job.fromJson(Map<String, dynamic> json) {
    return Job(
      jobId: json['job_id']?.toString() ?? '',
      jobTitle: json['job_title'] ?? '',
      company: json['company'],
      location: json['location'],
      category: json['category'],
      salary: json['salary'],
      descriptions: json['descriptions'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'job_id': jobId,
      'job_title': jobTitle,
      'company': company,
      'location': location,
      'category': category,
      'salary': salary,
      'descriptions': descriptions,
    };
  }
}

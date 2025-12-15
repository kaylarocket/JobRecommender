import 'package:flutter/foundation.dart';

import '../models/job.dart';
import '../models/recommendation.dart';
import '../services/api_service.dart';

class JobProvider extends ChangeNotifier {
  JobProvider(this._apiService);

  final ApiService _apiService;
  List<Job> jobs = [];
  List<Job> saved = [];
  List<Job> postedJobs = [];
  List<Map<String, dynamic>> applications = [];
  List<Recommendation> recommendations = [];
  bool isLoading = false;
  String? error;

  Future<void> loadJobs() async {
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      jobs = await _apiService.getJobs();
    } catch (e) {
      error = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  Future<void> refreshRecommendations(String userId) async {
    try {
      recommendations = await _apiService.getRecommendations(userId);
      notifyListeners();
    } catch (e) {
      // Keep UI usable even if recommendations fail.
      error = e.toString();
      notifyListeners();
    }
  }

  Future<void> saveJob(Job job) async {
    await _apiService.saveJob(job.jobId);
    if (!saved.any((j) => j.jobId == job.jobId)) {
      saved.add(job);
      notifyListeners();
    }
  }

  Future<void> apply(Job job, {String? coverLetter}) async {
    await _apiService.applyToJob(jobId: job.jobId, coverLetter: coverLetter);
    applications.add({
      'job_id': job.jobId,
      'job_title': job.jobTitle,
      'status': 'Submitted',
    });
    notifyListeners();
  }

  Future<Job> postJob({
    required String title,
    required String company,
    required String location,
    required String category,
    String? salary,
    required String description,
  }) async {
    final job = await _apiService.postJob(
      title: title,
      company: company,
      location: location,
      category: category,
      salary: salary,
      description: description,
    );
    jobs.insert(0, job);
    postedJobs.add(job);
    notifyListeners();
    return job;
  }
}

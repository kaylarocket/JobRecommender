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
  String? currentSearchQuery;
  String? currentCategory;

  bool isJobSaved(String jobId) {
    return saved.any((job) => job.jobId == jobId);
  }

  Future<void> loadJobs({
    String? query,
    String? location,
    String? category,
    String? sourceTag,
  }) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] loadJobs() START from source=$source, query=$query, category=$category');
    isLoading = true;
    error = null;
    currentSearchQuery = query;
    currentCategory = category;
    notifyListeners();
    try {
      jobs = await _apiService.getJobs(
        query: query,
        location: location,
        category: category,
        sourceTag: source,
      );
      print('[${DateTime.now()}] [JobProvider] loadJobs() END: ${jobs.length} jobs from source=$source');
    } catch (e) {
      print('[${DateTime.now()}] [JobProvider] loadJobs() ERROR from source=$source: $e');
      error = e.toString();
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }

  void clearSearch() {
    currentSearchQuery = null;
    currentCategory = null;
    loadJobs(sourceTag: 'clearSearch');
  }

  Future<void> refreshRecommendations(String userId, {String? sourceTag}) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] refreshRecommendations() START for user_id=$userId from source=$source');
    try {
      recommendations = await _apiService.getRecommendations(userId, sourceTag: source);
      print('[${DateTime.now()}] [JobProvider] refreshRecommendations() END: ${recommendations.length} recommendations from source=$source');
      notifyListeners();
    } catch (e) {
      print('[${DateTime.now()}] [JobProvider] refreshRecommendations() ERROR from source=$source: $e');
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

  Future<void> unsaveJob(Job job) async {
    await _apiService.unsaveJob(job.jobId);
    saved.removeWhere((j) => j.jobId == job.jobId);
    notifyListeners();
  }

  Future<bool> toggleSavedJob(Job job) async {
    if (isJobSaved(job.jobId)) {
      await unsaveJob(job);
      return false;
    }
    await saveJob(job);
    return true;
  }

  Future<void> apply(Job job, {String? coverLetter}) async {
    await _apiService.applyToJob(jobId: job.jobId, coverLetter: coverLetter);
    applications.insert(0, {
      'job_id': job.jobId,
      'job_title': job.jobTitle,
      'status': 'Submitted',
    });
    notifyListeners();
    await loadApplications(sourceTag: 'apply');
  }

  Future<void> loadSavedJobs({String? sourceTag}) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] loadSavedJobs() START from source=$source');
    try {
      saved = await _apiService.getSavedJobs(sourceTag: source);
      print('[${DateTime.now()}] [JobProvider] loadSavedJobs() END: ${saved.length} saved jobs from source=$source');
      notifyListeners();
    } catch (e) {
      print('[${DateTime.now()}] [JobProvider] loadSavedJobs() ERROR from source=$source: $e');
    }
  }

  Future<void> loadApplications({String? sourceTag}) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] loadApplications() START from source=$source');
    try {
      applications = await _apiService.getApplications(sourceTag: source);
      print('[${DateTime.now()}] [JobProvider] loadApplications() END: ${applications.length} applications from source=$source');
      notifyListeners();
    } catch (e) {
      print('[${DateTime.now()}] [JobProvider] loadApplications() ERROR from source=$source: $e');
    }
  }

  Future<List<Map<String, dynamic>>> fetchApplicantsForJob(
    String jobId, {
    int topN = 50,
    String? sourceTag,
  }) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] fetchApplicantsForJob() START for job_id=$jobId from source=$source');
    try {
      final applicants = await _apiService.getApplicantsForJob(
        jobId,
        topN: topN,
        sourceTag: source,
      );
      print('[${DateTime.now()}] [JobProvider] fetchApplicantsForJob() END: ${applicants.length} applicants for job_id=$jobId from source=$source');
      return applicants;
    } catch (e) {
      print('[${DateTime.now()}] [JobProvider] fetchApplicantsForJob() ERROR for job_id=$jobId from source=$source: $e');
      rethrow;
    }
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

  Future<void> loadPostedJobs({String? sourceTag}) async {
    final source = sourceTag ?? 'unknown';
    print('[${DateTime.now()}] [JobProvider] loadPostedJobs() START from source=$source');
    
    isLoading = true;
    error = null;
    notifyListeners();
    try {
      postedJobs = await _apiService.getRecruiterJobs(sourceTag: source);
      print('[${DateTime.now()}] [JobProvider] loadPostedJobs() END: ${postedJobs.length} jobs from source=$source');
    } catch (e) {
      error = e.toString();
      print('[${DateTime.now()}] [JobProvider] loadPostedJobs() ERROR: $error from source=$source');
    } finally {
      isLoading = false;
      notifyListeners();
    }
  }
}

import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/job.dart';
import '../models/recommendation.dart';
import '../models/user.dart';

/// API client that connects the Flutter app to the FastAPI backend (api_main.py).
/// - TF-IDF + LightFM hybrid scoring lives server-side; this client simply hits
///   /users/{id}/recommendations to retrieve the blended scores.
/// - Attach the JWT via Authorization header for endpoints that require it.
class ApiService {
  ApiService({this.baseUrl = 'http://localhost:8000'});

  final String baseUrl;
  String? _token;

  void updateToken(String? token) {
    _token = token;
  }

  Map<String, String> _headers({bool auth = false, String? source}) {
    return {
      'Content-Type': 'application/json',
      if (auth && _token != null) 'Authorization': 'Bearer $_token',
      if (source != null) 'X-Source': source,
    };
  }

  Future<UserSession> register({
    required String fullName,
    required String email,
    required String password,
    required String role,
    String? preferredLocation,
    String? headline,
    String? skills,
    int? experienceYears,
  }) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/auth/register'),
      headers: _headers(),
      body: jsonEncode({
        'full_name': fullName,
        'email': email,
        'password': password,
        'role': role,
        'preferred_location': preferredLocation,
        'headline': headline,
        'skills': skills,
        'experience_years': experienceYears,
      }),
    );
    return _parseAuthResponse(resp);
  }

  Future<UserSession> login({required String email, required String password}) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/auth/login'),
      headers: _headers(),
      body: jsonEncode({'email': email, 'password': password}),
    );
    return _parseAuthResponse(resp);
  }

  Future<List<Job>> getJobs({
    int page = 1,
    int pageSize = 40,
    String? query,
    String? location,
    String? category,
    String? sourceTag,
  }) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';
    
    final queryParams = {
      'page': page.toString(),
      'page_size': pageSize.toString(),
      if (query != null && query.isNotEmpty) 'query': query,
      if (location != null && location.isNotEmpty) 'location': location,
      if (category != null && category.isNotEmpty) 'category': category,
    };
    
    final uri = Uri.parse('$baseUrl/jobs').replace(queryParameters: queryParams);
    print('[${DateTime.now()}] [ApiService] GET /jobs request: url=$uri, source=$source');
    
    final resp = await http.get(uri, headers: _headers(source: source));
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;
    
    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /jobs response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to fetch jobs: ${resp.body}');
    }
    final decoded = jsonDecode(resp.body);
    final jobsCount = decoded is List ? decoded.length : (decoded['items'] as List).length;
    print('[${DateTime.now()}] [ApiService] GET /jobs response: status=${resp.statusCode}, jobs_count=$jobsCount, elapsed_ms=${elapsed}ms');
    
    if (decoded is List) {
      return decoded.map<Job>((e) => Job.fromJson(e as Map<String, dynamic>)).toList();
    }
    final items = (decoded['items'] as List).cast<Map<String, dynamic>>();
    return items.map(Job.fromJson).toList();
  }

  Future<Job> getJobDetails(String jobId) async {
    final resp = await http.get(
      Uri.parse('$baseUrl/jobs/$jobId'),
      headers: _headers(),
    );
    if (resp.statusCode != 200) {
      throw Exception('Job not found');
    }
    return Job.fromJson(jsonDecode(resp.body));
  }

  Future<List<Recommendation>> getRecommendations(String userId, {String? sourceTag}) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';
    
    print('[${DateTime.now()}] [ApiService] GET /recommendations request: user_id=$userId, source=$source');
    
    final resp = await http.get(
      Uri.parse('$baseUrl/users/$userId/recommendations'),
      headers: _headers(auth: true, source: source),
    );
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;
    
    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /recommendations response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to load recommendations');
    }
    final decoded = jsonDecode(resp.body) as List;
    print('[${DateTime.now()}] [ApiService] GET /recommendations response: status=${resp.statusCode}, recs_count=${decoded.length}, elapsed_ms=${elapsed}ms');
    
    return decoded.map((e) => Recommendation.fromJson(e)).toList();
  }

  Future<void> applyToJob({required String jobId, String? coverLetter}) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/applications'),
      headers: _headers(auth: true),
      body: jsonEncode({'job_id': jobId, 'cover_letter': coverLetter}),
    );
    if (resp.statusCode >= 300) {
      throw Exception('Failed to apply: ${resp.body}');
    }
  }

  Future<void> saveJob(String jobId) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/saved/$jobId'),
      headers: _headers(auth: true),
    );
    if (resp.statusCode >= 300) {
      throw Exception('Failed to save job');
    }
  }

  Future<void> unsaveJob(String jobId) async {
    final resp = await http.delete(
      Uri.parse('$baseUrl/saved/$jobId'),
      headers: _headers(auth: true),
    );
    if (resp.statusCode >= 300) {
      throw Exception('Failed to unsave job');
    }
  }

  Future<List<Job>> getSavedJobs({String? sourceTag}) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';

    print('[${DateTime.now()}] [ApiService] GET /saved request: source=$source');

    final resp = await http.get(
      Uri.parse('$baseUrl/saved'),
      headers: _headers(auth: true, source: source),
    );
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;

    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /saved response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to fetch saved jobs: ${resp.body}');
    }

    final decoded = jsonDecode(resp.body) as List;
    print('[${DateTime.now()}] [ApiService] GET /saved response: status=${resp.statusCode}, saved_count=${decoded.length}, elapsed_ms=${elapsed}ms');
    return decoded.map<Job>((e) => Job.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<List<Map<String, dynamic>>> getApplications({String? sourceTag}) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';

    print('[${DateTime.now()}] [ApiService] GET /applications request: source=$source');

    final resp = await http.get(
      Uri.parse('$baseUrl/applications'),
      headers: _headers(auth: true, source: source),
    );
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;

    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /applications response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to fetch applications: ${resp.body}');
    }

    final decoded = jsonDecode(resp.body) as List;
    print('[${DateTime.now()}] [ApiService] GET /applications response: status=${resp.statusCode}, app_count=${decoded.length}, elapsed_ms=${elapsed}ms');
    return decoded.cast<Map<String, dynamic>>();
  }

  Future<List<Map<String, dynamic>>> getApplicantsForJob(
    String jobId, {
    int topN = 50,
    String? sourceTag,
  }) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';

    final uri = Uri.parse('$baseUrl/employer/jobs/$jobId/recommendations')
        .replace(queryParameters: {'top_n': topN.toString()});
    print('[${DateTime.now()}] [ApiService] GET /employer/jobs/$jobId/recommendations request: source=$source');

    final resp = await http.get(
      uri,
      headers: _headers(auth: true, source: source),
    );
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;

    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /employer/jobs/$jobId/recommendations response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to fetch applicants: ${resp.body}');
    }

    final decoded = jsonDecode(resp.body) as List;
    print('[${DateTime.now()}] [ApiService] GET /employer/jobs/$jobId/recommendations response: status=${resp.statusCode}, applicant_count=${decoded.length}, elapsed_ms=${elapsed}ms');
    return decoded.cast<Map<String, dynamic>>();
  }

  Future<Job> postJob({
    required String title,
    required String company,
    required String location,
    required String category,
    String? salary,
    required String description,
  }) async {
    final resp = await http.post(
      Uri.parse('$baseUrl/jobs'),
      headers: _headers(auth: true),
      body: jsonEncode({
        'job_title': title,
        'company': company,
        'location': location,
        'category': category,
        'salary': salary,
        'descriptions': description,
      }),
    );
    if (resp.statusCode != 200) {
      throw Exception('Failed to post job: ${resp.body}');
    }
    return Job.fromJson(jsonDecode(resp.body));
  }

  Future<List<Job>> getRecruiterJobs({String? sourceTag}) async {
    final startTime = DateTime.now();
    final source = sourceTag ?? 'unknown';
    
    print('[${DateTime.now()}] [ApiService] GET /recruiter/jobs request: source=$source');
    
    final resp = await http.get(
      Uri.parse('$baseUrl/recruiter/jobs'),
      headers: _headers(auth: true, source: source),
    );
    final elapsed = DateTime.now().difference(startTime).inMilliseconds;
    
    if (resp.statusCode != 200) {
      print('[${DateTime.now()}] [ApiService] GET /recruiter/jobs response: status=${resp.statusCode}, ERROR, elapsed_ms=${elapsed}ms');
      throw Exception('Failed to fetch recruiter jobs: ${resp.body}');
    }
    final decoded = jsonDecode(resp.body) as List;
    print('[${DateTime.now()}] [ApiService] GET /recruiter/jobs response: status=${resp.statusCode}, jobs_count=${decoded.length}, elapsed_ms=${elapsed}ms');
    
    return decoded.map<Job>((e) => Job.fromJson(e as Map<String, dynamic>)).toList();
  }

  UserSession _parseAuthResponse(http.Response resp) {
    if (resp.statusCode != 200) {
      throw Exception('Authentication failed: ${resp.body}');
    }
    final decoded = jsonDecode(resp.body) as Map<String, dynamic>;
    final token = decoded['access_token'] as String;
    final user = UserProfile.fromJson(decoded['user'] as Map<String, dynamic>);
    updateToken(token);
    return UserSession(profile: user, token: token);
  }
}

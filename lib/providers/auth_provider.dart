import 'package:flutter/foundation.dart';

import '../models/user.dart';
import '../services/api_service.dart';

class AuthProvider extends ChangeNotifier {
  AuthProvider(this._apiService);

  final ApiService _apiService;
  UserSession? _session;
  bool _loading = false;
  String? _error;

  UserSession? get session => _session;
  bool get isLoading => _loading;
  String? get error => _error;
  bool get isAuthenticated => _session != null;
  bool get isRecruiter => _session?.profile.role == 'recruiter';

  Future<void> login(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _session = await _apiService.login(email: email, password: password);
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> register({
    required String fullName,
    required String email,
    required String password,
    required String role,
    String? preferredLocation,
    String? headline,
    String? skills,
    int? experienceYears,
  }) async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _session = await _apiService.register(
        fullName: fullName,
        email: email,
        password: password,
        role: role,
        preferredLocation: preferredLocation,
        headline: headline,
        skills: skills,
        experienceYears: experienceYears,
      );
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  void logout() {
    _session = null;
    _apiService.updateToken(null);
    notifyListeners();
  }

  // TODO: Persist session using SharedPreferences/secure storage for a real deployment.
}

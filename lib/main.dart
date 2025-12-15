import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'providers/auth_provider.dart';
import 'providers/job_provider.dart';
import 'services/api_service.dart';
import 'theme/app_theme.dart';
import 'pages/role_selection_page.dart';

void main() {
  final apiService = ApiService();
  runApp(JobRecommenderApp(apiService: apiService));
}

class JobRecommenderApp extends StatelessWidget {
  const JobRecommenderApp({super.key, required this.apiService});

  final ApiService apiService;

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AuthProvider(apiService)),
        ChangeNotifierProvider(create: (_) => JobProvider(apiService)),
      ],
      child: MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Hybrid Job Finder',
        theme: AppTheme.light(),
        home: const RoleSelectionPage(),
      ),
    );
  }
}

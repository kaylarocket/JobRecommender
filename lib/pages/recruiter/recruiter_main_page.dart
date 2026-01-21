import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import 'applicants_list_page.dart';
import 'applicants_page_new.dart';
import 'company_profile_page_new.dart';
import 'manage_jobs_page_new.dart';
import 'recruiter_dashboard_page_new.dart';

class RecruiterMainPage extends StatefulWidget {
  const RecruiterMainPage({super.key});

  @override
  State<RecruiterMainPage> createState() => _RecruiterMainPageState();
}

class _RecruiterMainPageState extends State<RecruiterMainPage> {
  int _currentIndex = 0;

  late final List<Widget> _pages = const [
    RecruiterDashboardPageNew(),
    ManageJobsPageNew(),
    ApplicantsListPage(),
    CompanyProfilePageNew(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: IndexedStack(
          index: _currentIndex,
          children: _pages,
        ),
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.08),
              blurRadius: 20,
              offset: const Offset(0, -4),
            ),
          ],
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildNavItem(Icons.dashboard_rounded, Icons.dashboard_outlined, 'Dashboard', 0),
                _buildNavItem(Icons.work_rounded, Icons.work_outline, 'Jobs', 1),
                _buildNavItem(Icons.people_rounded, Icons.people_outline, 'Applicants', 2),
                _buildNavItem(Icons.business_rounded, Icons.business_outlined, 'Company', 3),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNavItem(IconData activeIcon, IconData inactiveIcon, String label, int index) {
    final isActive = _currentIndex == index;
    return InkWell(
      onTap: () {
        if (index == 2) {
          final jobs = context.read<JobProvider>();
          jobs.loadPostedJobs(sourceTag: 'recruiter_nav');
          jobs.loadApplications(sourceTag: 'recruiter_nav');
        }
        setState(() => _currentIndex = index);
      },
      borderRadius: BorderRadius.circular(12),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: EdgeInsets.symmetric(
          horizontal: isActive ? 16 : 12,
          vertical: 8,
        ),
        decoration: BoxDecoration(
          color: isActive ? AppTheme.primary.withOpacity(0.1) : Colors.transparent,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              isActive ? activeIcon : inactiveIcon,
              color: isActive ? AppTheme.primary : Colors.black54,
              size: 24,
            ),
            const SizedBox(height: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: isActive ? FontWeight.w700 : FontWeight.w500,
                color: isActive ? AppTheme.primary : Colors.black54,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';

import 'auth/login_page.dart';

class RoleSelectionPage extends StatelessWidget {
  const RoleSelectionPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFFEEF2FF), Color(0xFFF8FAFC)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.workspaces_rounded, size: 56, color: Color(0xFF4F46E5)),
                const SizedBox(height: 16),
                const Text(
                  'Find or Post Jobs Effortlessly',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 10),
                const Text(
                  'Choose how you want to use the platform. The UI stays consistent with your role.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.black54),
                ),
                const SizedBox(height: 32),
                Row(
                  children: [
                    Expanded(child: _RoleCard(title: 'Job Seeker', description: 'Search, save, and apply to tailored roles.', icon: Icons.search_rounded, onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (_) => const LoginPage(initialRole: 'job_seeker')));
                    })),
                    const SizedBox(width: 16),
                    Expanded(child: _RoleCard(title: 'Recruiter', description: 'Post roles and view applicants quickly.', icon: Icons.badge_outlined, onTap: () {
                      Navigator.push(context, MaterialPageRoute(builder: (_) => const LoginPage(initialRole: 'recruiter')));
                    })),
                  ],
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _RoleCard extends StatelessWidget {
  const _RoleCard({required this.title, required this.description, required this.icon, required this.onTap});

  final String title;
  final String description;
  final IconData icon;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFE2E8F0)),
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 18, offset: const Offset(0, 10)),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFF4F46E5).withOpacity(0.1),
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: const Color(0xFF4F46E5)),
            ),
            const SizedBox(height: 12),
            Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
            const SizedBox(height: 8),
            Text(description, style: const TextStyle(color: Colors.black54)),
            const SizedBox(height: 10),
            const Row(
              children: [
                Text('Continue', style: TextStyle(color: Color(0xFF4F46E5), fontWeight: FontWeight.w700)),
                SizedBox(width: 6),
                Icon(Icons.arrow_forward_rounded, color: Color(0xFF4F46E5)),
              ],
            )
          ],
        ),
      ),
    );
  }
}

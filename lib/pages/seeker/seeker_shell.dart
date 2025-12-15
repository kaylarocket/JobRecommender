import 'package:flutter/material.dart';

import 'applications_page.dart';
import 'home_page.dart';
import 'profile_page.dart';
import 'saved_jobs_page.dart';

class SeekerShell extends StatefulWidget {
  const SeekerShell({super.key});

  @override
  State<SeekerShell> createState() => _SeekerShellState();
}

class _SeekerShellState extends State<SeekerShell> {
  int _index = 0;

  final pages = const [
    SeekerHomePage(),
    ApplicationsPage(),
    SavedJobsPage(),
    ProfilePage(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _index, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: (i) => setState(() => _index = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Home'),
          NavigationDestination(icon: Icon(Icons.file_present_outlined), selectedIcon: Icon(Icons.file_present), label: 'Applications'),
          NavigationDestination(icon: Icon(Icons.bookmark_outline), selectedIcon: Icon(Icons.bookmark), label: 'Saved'),
          NavigationDestination(icon: Icon(Icons.person_outline), selectedIcon: Icon(Icons.person), label: 'Profile'),
        ],
      ),
    );
  }
}

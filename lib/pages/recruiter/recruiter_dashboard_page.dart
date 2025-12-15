import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../widgets/job_card.dart';
import 'applicants_page.dart';
import 'post_job_page.dart';

class RecruiterDashboardPage extends StatelessWidget {
  const RecruiterDashboardPage({super.key});

  @override
  Widget build(BuildContext context) {
    final jobs = context.watch<JobProvider>();
    final profile = context.watch<AuthProvider>().session?.profile;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recruiter workspace'),
        actions: [
          IconButton(
            icon: const Icon(Icons.add_box_outlined),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const PostJobPage())),
          )
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Hi ${profile?.fullName.split(' ').first ?? 'Recruiter'}', style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
                const SizedBox(height: 4),
                const Text('Manage your openings and applicants', style: TextStyle(color: Colors.black54)),
              ]),
              ElevatedButton.icon(
                onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const PostJobPage())),
                icon: const Icon(Icons.edit),
                label: const Text('Post new job'),
              )
            ],
          ),
          const SizedBox(height: 16),
          Row(children: [
            Expanded(child: _statCard('Open roles', jobs.postedJobs.length.toString(), Icons.work_outline)),
            const SizedBox(width: 12),
            Expanded(child: _statCard('Applications', jobs.applications.length.toString(), Icons.inbox_outlined)),
          ]),
          const SizedBox(height: 16),
          const Text('My postings', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
          const SizedBox(height: 10),
          if (jobs.postedJobs.isEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), border: Border.all(color: const Color(0xFFE2E8F0))),
              child: const Text('No jobs posted yet. Tap "Post new job" to add one.'),
            )
          else
            ...jobs.postedJobs.map((job) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: JobCard(
                    job: job,
                    trailing: IconButton(
                      icon: const Icon(Icons.people_outline),
                      onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => ApplicantsPage(jobTitle: job.jobTitle))),
                    ),
                  ),
                ))
        ],
      ),
    );
  }

  Widget _statCard(String label, String value, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: const Color(0xFFEEF2FF), borderRadius: BorderRadius.circular(12)),
            child: Icon(icon, color: const Color(0xFF4F46E5)),
          ),
          const SizedBox(width: 12),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(label, style: const TextStyle(color: Colors.black54)),
            Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
          ])
        ],
      ),
    );
  }
}

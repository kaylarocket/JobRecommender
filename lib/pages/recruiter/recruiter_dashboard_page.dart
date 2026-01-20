import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../widgets/job_card.dart';
import 'applicants_page.dart';
import 'post_job_page.dart';

class RecruiterDashboardPage extends StatefulWidget {
  const RecruiterDashboardPage({super.key});

  @override
  State<RecruiterDashboardPage> createState() => _RecruiterDashboardPageState();
}

class _RecruiterDashboardPageState extends State<RecruiterDashboardPage> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      print('[${DateTime.now()}] [recruiter_dashboard_page] addPostFrameCallback fired');
      print('[${DateTime.now()}] [recruiter_dashboard_page] calling loadPostedJobs() from source=recruiter_dashboard');
      context.read<JobProvider>().loadPostedJobs(sourceTag: 'recruiter_dashboard');
      context.read<JobProvider>().loadApplications(sourceTag: 'recruiter_dashboard');
    });
  }

  @override
  Widget build(BuildContext context) {
    final jobs = context.watch<JobProvider>();
    final profile = context.watch<AuthProvider>().session?.profile;
    final rawName = profile?.fullName.trim() ?? '';
    final firstName = rawName.isEmpty ? 'Recruiter' : rawName.split(' ').first;
    final recentApplicants = jobs.applications;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text('Welcome back, $firstName', style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
        const SizedBox(height: 4),
        const Text('Manage your postings and applicants', style: TextStyle(color: Colors.black54)),
        const SizedBox(height: 16),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () => _openPostJob(context),
            icon: const Icon(Icons.edit_outlined),
            label: const Text('Post a Job'),
          ),
        ),
        const SizedBox(height: 16),
        LayoutBuilder(
          builder: (context, constraints) {
            if (constraints.maxWidth < 360) {
              return Column(
                children: [
                  _statCard('Open roles', jobs.postedJobs.length.toString(), Icons.work_outline),
                  const SizedBox(height: 12),
                  _statCard('Applications', jobs.applications.length.toString(), Icons.inbox_outlined),
                ],
              );
            }
            return Row(children: [
              Expanded(child: _statCard('Open roles', jobs.postedJobs.length.toString(), Icons.work_outline)),
              const SizedBox(width: 12),
              Expanded(child: _statCard('Applications', jobs.applications.length.toString(), Icons.inbox_outlined)),
            ]);
          },
        ),
        const SizedBox(height: 24),
        const Text('Posted Jobs', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
        const SizedBox(height: 12),
        if (jobs.isLoading)
          const Center(child: CircularProgressIndicator())
        else if (jobs.postedJobs.isEmpty)
          _emptyState(
            title: 'No posted jobs yet',
            subtitle: 'Create your first role to start receiving applicants.',
            icon: Icons.work_outline,
          )
        else
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: jobs.postedJobs.length,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, index) {
              final job = jobs.postedJobs[index];
              return JobCard(
                job: job,
                trailing: IconButton(
                  tooltip: 'View applicants',
                  icon: const Icon(Icons.people_outline),
                  onPressed: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => ApplicantsPage(jobTitle: job.jobTitle, jobId: job.jobId)),
                  ),
                ),
              );
            },
          ),
        const SizedBox(height: 24),
        const Text('Recent Applicants', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
        const SizedBox(height: 12),
        if (recentApplicants.isEmpty)
          _emptyState(
            title: 'No applicants yet',
            subtitle: 'Applicants will appear once roles are live.',
            icon: Icons.people_outline,
          )
        else
          ListView.separated(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: recentApplicants.length,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, index) {
              final applicant = recentApplicants[index];
              return _applicantCard(applicant);
            },
          ),
      ],
    );
  }

  void _openPostJob(BuildContext context) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => const PostJobPage()));
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

  Widget _emptyState({required String title, required String subtitle, required IconData icon}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFF1F5F9),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: Colors.black54),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(color: Colors.black54),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _applicantCard(Map<String, dynamic> applicant) {
    final roleTitle = (applicant['job_title'] ?? 'Job').toString();
    final status = (applicant['status'] ?? 'Submitted').toString();
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          const CircleAvatar(
            radius: 20,
            backgroundColor: Color(0xFFEEF2FF),
            child: Icon(Icons.person_outline, color: Color(0xFF4F46E5)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(roleTitle, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w700)),
                const SizedBox(height: 4),
                Text('Status: $status', style: const TextStyle(color: Colors.black54)),
              ],
            ),
          ),
          const Icon(Icons.chevron_right_rounded, color: Colors.black45),
        ],
      ),
    );
  }
}

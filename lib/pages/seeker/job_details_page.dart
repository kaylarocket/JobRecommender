import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/job.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/primary_button.dart';

class JobDetailsPage extends StatelessWidget {
  const JobDetailsPage({super.key, required this.job});

  final Job job;

  @override
  Widget build(BuildContext context) {
    final jobProvider = context.watch<JobProvider>();
    final isSaved = jobProvider.isJobSaved(job.jobId);
    return Scaffold(
      appBar: AppBar(title: const Text('Job details')),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(job.jobTitle,
                            style: const TextStyle(
                                fontSize: 22, fontWeight: FontWeight.w800)),
                        const SizedBox(height: 6),
                        Text(job.company ?? '',
                            style: const TextStyle(
                                color: Colors.black54,
                                fontWeight: FontWeight.w700)),
                        const SizedBox(height: 12),
                        Wrap(spacing: 8, runSpacing: 8, children: [
                          _chip(Icons.place, job.location ?? 'Remote'),
                          _chip(Icons.category, job.category ?? ''),
                          _chip(
                              Icons.payments,
                              job.salary?.isNotEmpty == true
                                  ? job.salary!
                                  : 'Not disclosed'),
                        ]),
                      ],
                    ),
                  ),
                  IconButton(
                    icon: Icon(isSaved ? Icons.bookmark_rounded : Icons.bookmark_border_rounded),
                    onPressed: () async {
                      final nowSaved = await jobProvider.toggleSavedJob(job);
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(nowSaved ? 'Saved job' : 'Removed from saved jobs'),
                        ),
                      );
                    },
                  )
                ],
              ),
              const SizedBox(height: 20),
              Text(job.descriptions ?? 'No description provided',
                  style: const TextStyle(height: 1.5)),
              const SizedBox(height: 20),
              Row(
                children: [
                  Expanded(
                    child: PrimaryButton(
                      label: 'Apply',
                      onPressed: () async {
                        await jobProvider.apply(job);
                        ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                                content: Text('Application submitted')));
                      },
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () async {
                        final nowSaved = await jobProvider.toggleSavedJob(job);
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(nowSaved ? 'Saved for later' : 'Removed from saved jobs'),
                          ),
                        );
                      },
                      style: OutlinedButton.styleFrom(
                        foregroundColor: AppTheme.primary,
                        side: const BorderSide(color: AppTheme.primary),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16)),
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                      child: Text(
                        isSaved ? 'Unsave job' : 'Save job',
                        style: const TextStyle(fontWeight: FontWeight.w700),
                      ),
                    ),
                  )
                ],
              ),
              const SizedBox(height: 16),
              const Text(
                  'Note: applications and saved jobs sync with the server and will appear after you sign back in.',
                  style: TextStyle(color: Colors.black54, fontSize: 12)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _chip(IconData icon, String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF1F5F9),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.black54),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';
import '../../widgets/job_card.dart';
import 'job_details_page.dart';

class SavedJobsPage extends StatelessWidget {
  const SavedJobsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final saved = context.watch<JobProvider>().saved;
    if (saved.isEmpty) {
      return const Center(child: Text('Save jobs you like to track them here.'));
    }
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: saved.length,
      itemBuilder: (context, index) {
        final job = saved[index];
        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: JobCard(
            job: job,
            onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => JobDetailsPage(job: job))),
          ),
        );
      },
    );
  }
}

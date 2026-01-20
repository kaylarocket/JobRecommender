import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';

class ApplicationsPage extends StatelessWidget {
  const ApplicationsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final applications = context.watch<JobProvider>().applications;
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: applications.length,
      itemBuilder: (context, index) {
        final app = applications[index];
        final status = _formatStatus(app['status']);
        return Container(
          margin: const EdgeInsets.only(bottom: 12),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: const Color(0xFFE2E8F0)),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(app['job_title'] ?? 'Job', style: const TextStyle(fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  Text('Status: $status', style: const TextStyle(color: Colors.black54)),
                ],
              ),
              const Icon(Icons.chevron_right_rounded)
            ],
          ),
        );
      },
    );
  }

  String _formatStatus(dynamic value) {
    final raw = (value ?? '').toString().trim();
    if (raw.isEmpty) {
      return 'Submitted';
    }
    final normalized = raw.toLowerCase();
    if (normalized == 'scheduled') {
      return 'Scheduled interview';
    }
    if (normalized == 'rejected') {
      return 'Rejected';
    }
    if (normalized == 'submitted') {
      return 'Submitted';
    }
    return raw[0].toUpperCase() + raw.substring(1);
  }
}

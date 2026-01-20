import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';

class ApplicantsPage extends StatefulWidget {
  const ApplicantsPage({super.key, required this.jobTitle, required this.jobId});

  final String jobTitle;
  final String jobId;

  @override
  State<ApplicantsPage> createState() => _ApplicantsPageState();
}

class _ApplicantsPageState extends State<ApplicantsPage> {
  late Future<List<Map<String, dynamic>>> _futureApplicants;

  @override
  void initState() {
    super.initState();
    _futureApplicants = context.read<JobProvider>().fetchApplicantsForJob(
          widget.jobId,
          sourceTag: 'applicants_page',
        );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Applicants • ${widget.jobTitle}',
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
        ),
      ),
      body: SafeArea(
        top: false,
        child: FutureBuilder<List<Map<String, dynamic>>>(
          future: _futureApplicants,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return _errorState(snapshot.error.toString());
            }
            final applicants = snapshot.data ?? [];
            if (applicants.isEmpty) {
              return _emptyState();
            }
            return ListView.separated(
              padding: const EdgeInsets.all(16),
              itemCount: applicants.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (context, index) {
                final applicant = applicants[index];
                final name = (applicant['full_name'] ?? '').toString().trim();
                final status = _formatStatus(applicant['status'] ?? 'Submitted');
                final headline = (applicant['headline'] ?? '').toString().trim();
                final scoreValue = applicant['score'];
                final score = scoreValue is num ? scoreValue.toDouble() : 0.0;
                final matchPercent = (score * 100).clamp(0, 100).round();
                final applicationId = (applicant['application_id'] ?? '').toString();
                return _applicantCard(
                  name: name.isEmpty ? 'Candidate' : name,
                  status: status,
                  headline: headline,
                  matchPercent: matchPercent,
                  applicationId: applicationId,
                  onStatusChange: (value) => _updateApplicantStatus(applicationId, value),
                );
              },
            );
          },
        ),
      ),
    );
  }

  Widget _applicantCard({
    required String name,
    required String status,
    required String headline,
    required int matchPercent,
    required String applicationId,
    required void Function(String) onStatusChange,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFFE2E8F0)),
      ),
      child: Row(
        children: [
          const CircleAvatar(
            radius: 18,
            backgroundColor: Color(0xFFEEF2FF),
            child: Icon(Icons.person_outline, color: Color(0xFF4F46E5)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w800)),
                const SizedBox(height: 4),
                if (headline.isNotEmpty)
                  Text(headline, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(color: Colors.black54)),
                if (headline.isNotEmpty) const SizedBox(height: 2),
                Text('Status: $status', style: const TextStyle(color: Colors.black54)),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('$matchPercent%', style: const TextStyle(fontWeight: FontWeight.w800)),
              const Text('Match', style: TextStyle(color: Colors.black54, fontSize: 12)),
              if (applicationId.isNotEmpty)
                PopupMenuButton<String>(
                  icon: const Icon(Icons.more_horiz, color: Colors.black45),
                  onSelected: onStatusChange,
                  itemBuilder: (context) => const [
                    PopupMenuItem(
                      value: 'scheduled',
                      child: Text('Schedule interview'),
                    ),
                    PopupMenuItem(
                      value: 'rejected',
                      child: Text('Reject applicant'),
                    ),
                    PopupMenuItem(
                      value: 'submitted',
                      child: Text('Mark as submitted'),
                    ),
                  ],
                ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _updateApplicantStatus(String applicationId, String status) async {
    if (applicationId.isEmpty) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Missing application ID')),
      );
      return;
    }
    try {
      await context.read<JobProvider>().updateApplicationStatus(
            applicationId,
            status,
            sourceTag: 'applicants_page',
          );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Applicant marked as ${_formatStatus(status)}')),
      );
      setState(() {
        _futureApplicants = context.read<JobProvider>().fetchApplicantsForJob(
              widget.jobId,
              sourceTag: 'applicants_page_refresh',
            );
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to update status: $e')),
      );
    }
  }

  String _formatStatus(String value) {
    final raw = value.trim();
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

  Widget _emptyState() {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
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
                child: const Icon(Icons.people_outline, color: Colors.black54),
              ),
              const SizedBox(width: 12),
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('No applicants yet', style: TextStyle(fontWeight: FontWeight.w700)),
                    SizedBox(height: 4),
                    Text('Applicants will appear here as candidates apply.', style: TextStyle(color: Colors.black54)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _errorState(String message) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
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
                  color: const Color(0xFFFEE2E2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.error_outline, color: Color(0xFFDC2626)),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Unable to load applicants', style: TextStyle(fontWeight: FontWeight.w700)),
                    const SizedBox(height: 4),
                    Text(message, style: const TextStyle(color: Colors.black54)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

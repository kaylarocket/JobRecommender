import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';

class ApplicantsListPage extends StatefulWidget {
  const ApplicantsListPage({super.key});

  @override
  State<ApplicantsListPage> createState() => _ApplicantsListPageState();
}

class _ApplicantsListPageState extends State<ApplicantsListPage> {
  String _selectedStatus = 'All';
  late Future<void> _loadFuture;

  @override
  void initState() {
    super.initState();
    _loadFuture = context.read<JobProvider>().loadApplications(sourceTag: 'applicants_list');
  }

  @override
  Widget build(BuildContext context) {
    final applications = context.watch<JobProvider>().applications;
    final statuses = _buildStatusFilters(applications);
    final selectedStatus = statuses.contains(_selectedStatus) ? _selectedStatus : 'All';
    final filteredApplicants = _filterApplicants(applications, selectedStatus);

    return FutureBuilder<void>(
      future: _loadFuture,
      builder: (context, snapshot) {
        final isLoading = snapshot.connectionState == ConnectionState.waiting;
        return Column(
          children: [
            // Title
            Padding(
              padding: const EdgeInsets.all(16),
              child: const Row(
                children: [
                  Text(
                    'Applicants',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.w800),
                  ),
                ],
              ),
            ),
            // Filter
            Container(
              color: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: statuses
                      .map((status) => Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: FilterChip(
                              label: Text(status),
                              selected: selectedStatus == status,
                              onSelected: (_) => setState(() => _selectedStatus = status),
                            ),
                          ))
                      .toList(),
                ),
              ),
            ),
            const Divider(height: 1),
            // List
            Expanded(
              child: isLoading && applications.isEmpty
                  ? const Center(child: CircularProgressIndicator())
                  : filteredApplicants.isEmpty
                      ? const Center(child: Text('No applicants'))
                      : ListView.separated(
                          padding: const EdgeInsets.all(16),
                          itemCount: filteredApplicants.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 12),
                          itemBuilder: (context, index) {
                            final applicant = filteredApplicants[index];
                            final name = _formatName(applicant['applicant_name']);
                            final jobTitle = _formatJobTitle(applicant['job_title']);
                            final status = _formatStatus(applicant['status']);
                            final appliedAt = _formatAppliedDate(applicant['created_at']);
                            return Container(
                              padding: const EdgeInsets.all(14),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(14),
                                border: Border.all(color: const Color(0xFFE2E8F0)),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    children: [
                                      CircleAvatar(
                                        radius: 20,
                                        backgroundColor: const Color(0xFFEEF2FF),
                                        child: Text(
                                          name.isNotEmpty ? name[0].toUpperCase() : 'C',
                                          style: const TextStyle(color: Color(0xFF4F46E5)),
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: Column(
                                          crossAxisAlignment: CrossAxisAlignment.start,
                                          children: [
                                            Text(
                                              name,
                                              style: const TextStyle(fontWeight: FontWeight.w700),
                                            ),
                                            const SizedBox(height: 4),
                                            Text(
                                              jobTitle,
                                              style: const TextStyle(fontSize: 12, color: Colors.black54),
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                            ),
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                    children: [
                                      Text('Status: $status', style: const TextStyle(color: Colors.black54)),
                                      PopupMenuButton<String>(
                                        icon: const Icon(Icons.more_horiz, color: Colors.black45),
                                        onSelected: (value) => _updateApplicantStatus(
                                          context,
                                          applicant['id']?.toString() ?? '',
                                          value,
                                        ),
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
                                      Text(appliedAt, style: const TextStyle(color: Colors.black54, fontSize: 12)),
                                    ],
                                  ),
                                ],
                              ),
                            );
                          },
                        ),
            ),
          ],
        );
      },
    );
  }

  List<String> _buildStatusFilters(List<Map<String, dynamic>> applications) {
    final seen = <String>{};
    for (final app in applications) {
      final status = _formatStatus(app['status']);
      if (status.isNotEmpty) {
        seen.add(status);
      }
    }
    final sorted = seen.toList()..sort();
    return ['All', ...sorted];
  }

  List<Map<String, dynamic>> _filterApplicants(
    List<Map<String, dynamic>> applications,
    String statusFilter,
  ) {
    final sorted = List<Map<String, dynamic>>.from(applications);
    sorted.sort((a, b) => _parseDate(b['created_at']).compareTo(_parseDate(a['created_at'])));
    if (statusFilter == 'All') {
      return sorted;
    }
    return sorted.where((app) => _formatStatus(app['status']) == statusFilter).toList();
  }

  DateTime _parseDate(dynamic value) {
    if (value == null) {
      return DateTime.fromMillisecondsSinceEpoch(0);
    }
    if (value is DateTime) {
      return value;
    }
    final parsed = DateTime.tryParse(value.toString());
    return parsed ?? DateTime.fromMillisecondsSinceEpoch(0);
  }

  String _formatAppliedDate(dynamic value) {
    final parsed = _parseDate(value);
    if (parsed.year <= 1970) {
      return 'Applied date unavailable';
    }
    final local = parsed.toLocal();
    final month = local.month.toString().padLeft(2, '0');
    final day = local.day.toString().padLeft(2, '0');
    return '${local.year}-$month-$day';
  }

  String _formatName(dynamic value) {
    final name = (value ?? '').toString().trim();
    return name.isEmpty ? 'Candidate' : name;
  }

  String _formatJobTitle(dynamic value) {
    final title = (value ?? '').toString().trim();
    return title.isEmpty ? 'Job' : title;
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

  Future<void> _updateApplicantStatus(
    BuildContext context,
    String applicationId,
    String status,
  ) async {
    if (applicationId.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Missing application ID')),
      );
      return;
    }
    try {
      await context.read<JobProvider>().updateApplicationStatus(
            applicationId,
            status,
            sourceTag: 'applicants_list',
          );
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Applicant marked as ${_formatStatus(status)}')),
      );
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to update status: $e')),
      );
    }
  }
}

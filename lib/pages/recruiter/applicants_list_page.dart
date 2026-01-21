import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';

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
    _loadFuture = _loadApplicants(sourceTag: 'applicants_list');
  }

  @override
  Widget build(BuildContext context) {
    final applications = context.watch<JobProvider>().applications;
    final postedJobs = context.watch<JobProvider>().postedJobs;
    
    // Get job IDs posted by this recruiter
    final recruiterJobIds = postedJobs.map((job) => job.jobId).toSet();
    
    // Filter applications to only those for recruiter's jobs
    final recruiterApplications = applications
        .where((app) {
          final jobId = app['job_id']?.toString() ?? '';
          return recruiterJobIds.contains(jobId);
        })
        .toList();
    
    final statuses = _buildStatusFilters(recruiterApplications);
    final selectedStatus = statuses.contains(_selectedStatus) ? _selectedStatus : 'All';
    final filteredApplicants = _filterApplicants(recruiterApplications, selectedStatus);

    return FutureBuilder<void>(
      future: _loadFuture,
      builder: (context, snapshot) {
        final isLoading = snapshot.connectionState == ConnectionState.waiting;
        return Scaffold(
          backgroundColor: const Color(0xFFF8F9FA),
          body: RefreshIndicator(
            onRefresh: () => _loadApplicants(sourceTag: 'applicants_list_refresh'),
            child: CustomScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              slivers: [
              // Beautiful Gradient Header
              SliverAppBar(
                backgroundColor: Colors.transparent,
                elevation: 0,
                pinned: true,
                expandedHeight: 160,
                flexibleSpace: FlexibleSpaceBar(
                  background: Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          AppTheme.primary,
                          AppTheme.accent,
                        ],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      borderRadius: const BorderRadius.only(
                        bottomLeft: Radius.circular(32),
                        bottomRight: Radius.circular(32),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.primary.withOpacity(0.3),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: SafeArea(
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(24, 40, 24, 24),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.2),
                                    borderRadius: BorderRadius.circular(16),
                                  ),
                                  child: const Icon(
                                    Icons.people_rounded,
                                    color: Colors.white,
                                    size: 28,
                                  ),
                                ),
                                const SizedBox(width: 16),
                                Expanded(
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      const Text(
                                        'All Applicants',
                                        style: TextStyle(
                                          fontSize: 28,
                                          fontWeight: FontWeight.w800,
                                          color: Colors.white,
                                          letterSpacing: -0.5,
                                        ),
                                      ),
                                      const SizedBox(height: 4),
                                      Text(
                                        'View applications across all jobs',
                                        style: TextStyle(
                                          fontSize: 14,
                                          color: Colors.white.withOpacity(0.9),
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),

              // Filter Chips
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: statuses
                          .map((status) => Padding(
                                padding: const EdgeInsets.only(right: 8),
                                child: FilterChip(
                                  label: Text(status),
                                  selected: selectedStatus == status,
                                  onSelected: (_) =>
                                      setState(() => _selectedStatus = status),
                                  backgroundColor: Colors.white,
                                  selectedColor:
                                      AppTheme.accent.withOpacity(0.2),
                                  side: BorderSide(
                                    color: selectedStatus == status
                                        ? AppTheme.accent
                                        : AppTheme.divider,
                                  ),
                                  labelStyle: TextStyle(
                                    color: selectedStatus == status
                                        ? AppTheme.accent
                                        : Colors.black87,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ))
                          .toList(),
                    ),
                  ),
                ),
              ),

              // Content
              if (isLoading && applications.isEmpty)
                const SliverToBoxAdapter(
                  child: Padding(
                    padding: EdgeInsets.all(40),
                    child: CircularProgressIndicator(),
                  ),
                )
              else if (filteredApplicants.isEmpty)
                SliverToBoxAdapter(
                  child: Padding(
                    padding: const EdgeInsets.all(40),
                    child: Column(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(24),
                          decoration: BoxDecoration(
                            color: AppTheme.muted,
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            Icons.people_outline,
                            size: 48,
                            color: AppTheme.mutedText,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'No applicants',
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(
                                fontWeight: FontWeight.w700,
                              ),
                        ),
                      ],
                    ),
                  ),
                )
              else
                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  sliver: SliverList(
                    delegate: SliverChildBuilderDelegate(
                      (context, index) {
                        final applicant = filteredApplicants[index];
                        final name = (applicant['applicant_name'] ??
                                applicant['full_name'] ??
                                '')
                            .toString()
                            .trim();
                        final status = _formatStatus(
                            applicant['status'] ?? 'Submitted');
                        final headline = (applicant['applicant_headline'] ??
                                applicant['headline'] ??
                                '')
                            .toString()
                            .trim();
                        final scoreValue = applicant['score'];
                        final score = scoreValue is num
                            ? scoreValue.toDouble()
                            : 0.0;
                        final matchPercent =
                            (score * 100).clamp(0, 100).round();
                        final jobTitle = (applicant['job_title'] ?? '')
                            .toString()
                            .trim();

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 16),
                          child: _buildApplicantCard(
                            name: name.isEmpty ? 'Candidate' : name,
                            status: status,
                            headline: headline,
                            matchPercent: matchPercent,
                            jobTitle: jobTitle,
                          ),
                        );
                      },
                      childCount: filteredApplicants.length,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _loadApplicants({String sourceTag = 'applicants_list'}) async {
    final jobs = context.read<JobProvider>();
    await jobs.loadPostedJobs(sourceTag: sourceTag);
    await jobs.loadApplications(sourceTag: sourceTag);
  }

  Widget _buildApplicantCard({
    required String name,
    required String status,
    required String headline,
    required int matchPercent,
    required String jobTitle,
  }) {
    final statusColor = _getStatusColor(status);

    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 16,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: () {
            // TODO: Navigate to applicant details
          },
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header: Avatar, Name and Match Score
                Row(
                  children: [
                    // Modern Avatar with gradient
                    Container(
                      width: 56,
                      height: 56,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            AppTheme.primary.withOpacity(0.2),
                            AppTheme.accent.withOpacity(0.2),
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: AppTheme.primary.withOpacity(0.1),
                          width: 2,
                        ),
                      ),
                      child: Icon(
                        Icons.person_rounded,
                        color: AppTheme.primary,
                        size: 28,
                      ),
                    ),
                    const SizedBox(width: 16),
                    // Name and Headline
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            name,
                            style: Theme.of(context)
                                .textTheme
                                .titleMedium
                                ?.copyWith(
                                  fontWeight: FontWeight.w800,
                                  fontSize: 17,
                                ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          if (headline.isNotEmpty) ...[
                            const SizedBox(height: 4),
                            Row(
                              children: [
                                Icon(
                                  Icons.work_outline_rounded,
                                  size: 14,
                                  color: AppTheme.mutedText,
                                ),
                                const SizedBox(width: 4),
                                Expanded(
                                  child: Text(
                                    headline,
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodySmall
                                        ?.copyWith(
                                          color: AppTheme.mutedText,
                                          fontSize: 13,
                                        ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ],
                      ),
                    ),
                    const SizedBox(width: 12),
                    // Match Score Badge with gradient
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [
                            AppTheme.primary,
                            AppTheme.accent,
                          ],
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        boxShadow: [
                          BoxShadow(
                            color: AppTheme.primary.withOpacity(0.3),
                            blurRadius: 8,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          Text(
                            '$matchPercent%',
                            style: const TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w900,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Match',
                            style: TextStyle(
                              fontSize: 10,
                              color: Colors.white.withOpacity(0.9),
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                // Divider
                Container(
                  height: 1,
                  decoration: BoxDecoration(
                    gradient: LinearGradient(
                      colors: [
                        Colors.transparent,
                        AppTheme.divider,
                        Colors.transparent,
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                // Job Title and Status Row
                Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          if (jobTitle.isNotEmpty)
                            Text(
                              jobTitle,
                              style: Theme.of(context)
                                  .textTheme
                                  .bodySmall
                                  ?.copyWith(
                                    color: AppTheme.mutedText,
                                    fontSize: 12,
                                  ),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 12),
                    // Status Badge with icon
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: statusColor.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: statusColor.withOpacity(0.3),
                          width: 1.5,
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            _getStatusIcon(status),
                            size: 16,
                            color: statusColor,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            status,
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: statusColor,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
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

  String _formatStatus(dynamic value) {
    final raw = (value ?? '').toString().trim();
    if (raw.isEmpty) {
      return 'Submitted';
    }
    final normalized = raw.toLowerCase();
    if (normalized == 'scheduled') {
      return 'Scheduled';
    }
    if (normalized == 'rejected') {
      return 'Rejected';
    }
    if (normalized == 'accepted') {
      return 'Accepted';
    }
    if (normalized == 'submitted') {
      return 'Submitted';
    }
    return raw[0].toUpperCase() + raw.substring(1);
  }

  Color _getStatusColor(String status) {
    switch (status.toLowerCase()) {
      case 'accepted':
        return const Color(0xFF10B981);
      case 'scheduled':
        return const Color(0xFF3B82F6);
      case 'rejected':
        return const Color(0xFFEF4444);
      default:
        return const Color(0xFF64748B);
    }
  }

  IconData _getStatusIcon(String status) {
    switch (status.toLowerCase()) {
      case 'accepted':
        return Icons.check_circle_rounded;
      case 'scheduled':
        return Icons.calendar_today_rounded;
      case 'rejected':
        return Icons.cancel_rounded;
      default:
        return Icons.send_rounded;
    }
  }
}

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import 'post_job_page.dart';

class ManageJobsPageNew extends StatefulWidget {
  const ManageJobsPageNew({super.key});

  @override
  State<ManageJobsPageNew> createState() => _ManageJobsPageNewState();
}

class _ManageJobsPageNewState extends State<ManageJobsPageNew> {
  String _filterStatus = 'All';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<JobProvider>().loadPostedJobs(sourceTag: 'manage_jobs_new');
    });
  }

  @override
  Widget build(BuildContext context) {
    final jobs = context.watch<JobProvider>();
    final filteredJobs = _filterStatus == 'All'
        ? jobs.postedJobs
        : jobs.postedJobs.where((job) => _matchesStatus(job, _filterStatus)).toList();

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          // Modern App Bar
          SliverAppBar(
            backgroundColor: AppTheme.accent,
            elevation: 0,
            pinned: true,
            expandedHeight: 180,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [AppTheme.accent, AppTheme.primary],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(20, 60, 20, 20),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        Row(
                          children: [
                            Text(
                              'Manage Jobs',
                              style: const TextStyle(
                                fontSize: 28,
                                fontWeight: FontWeight.w800,
                                color: Colors.white,
                              ),
                            ),
                            const Spacer(),
                            if (jobs.postedJobs.isNotEmpty)
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 6,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.white,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  '${jobs.postedJobs.length}',
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w700,
                                    color: AppTheme.accent,
                                  ),
                                ),
                              ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'View and manage all your job postings',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.white.withOpacity(0.85),
                          ),
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
                  children: ['All', 'Active', 'Closed']
                      .map((status) => Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: FilterChip(
                              label: Text(status),
                              selected: _filterStatus == status,
                              onSelected: (selected) {
                                setState(() => _filterStatus = status);
                              },
                              backgroundColor: Colors.white,
                              selectedColor: AppTheme.accent.withOpacity(0.2),
                              side: BorderSide(
                                color: _filterStatus == status
                                    ? AppTheme.accent
                                    : AppTheme.divider,
                              ),
                              labelStyle: TextStyle(
                                color: _filterStatus == status
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

          // Jobs List
          if (jobs.isLoading)
            const SliverToBoxAdapter(
              child: Padding(
                padding: EdgeInsets.all(40),
                child: CircularProgressIndicator(),
              ),
            )
          else if (filteredJobs.isEmpty)
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
                        Icons.work_outline,
                        size: 48,
                        color: AppTheme.mutedText,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'No jobs found',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Post a new job to get started',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.mutedText,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    ElevatedButton.icon(
                      onPressed: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const PostJobPage()),
                      ),
                      icon: const Icon(Icons.add),
                      label: const Text('Post a Job'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.accent,
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
                    final job = filteredJobs[index];
                    final statusValue = _normalizeStatus(job.status);

                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: AppTheme.divider),
                        boxShadow: AppTheme.boxShadowSmall,
                      ),
                      child: Material(
                        color: Colors.transparent,
                        child: InkWell(
                          borderRadius: BorderRadius.circular(16),
                          onTap: () {
                            // TODO: Navigate to job details/edit
                          },
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            job.jobTitle,
                                            style: Theme.of(context)
                                                .textTheme
                                                .titleMedium
                                                ?.copyWith(
                                                  fontWeight: FontWeight.w700,
                                                ),
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                          const SizedBox(height: 4),
                                          Text(
                                            job.location ?? 'Remote',
                                            style: Theme.of(context)
                                                .textTheme
                                                .bodySmall
                                                ?.copyWith(
                                                  color: AppTheme.mutedText,
                                                ),
                                            maxLines: 1,
                                            overflow: TextOverflow.ellipsis,
                                          ),
                                        ],
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Container(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 10,
                                        vertical: 6,
                                      ),
                                      decoration: BoxDecoration(
                                        color: statusValue == 'active'
                                            ? AppTheme.success.withOpacity(0.1)
                                            : AppTheme.error.withOpacity(0.1),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: Text(
                                        statusValue == 'active' ? 'ACTIVE' : 'CLOSED',
                                        style: TextStyle(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w700,
                                          color: statusValue == 'active'
                                              ? AppTheme.success
                                              : AppTheme.error,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                Row(
                                  children: [
                                    DropdownButtonHideUnderline(
                                      child: DropdownButton<String>(
                                        value: statusValue,
                                        items: const [
                                          DropdownMenuItem(
                                            value: 'active',
                                            child: Text('Active'),
                                          ),
                                          DropdownMenuItem(
                                            value: 'closed',
                                            child: Text('Closed'),
                                          ),
                                        ],
                                        onChanged: (value) async {
                                          if (value == null || value == statusValue) {
                                            return;
                                          }
                                          try {
                                            await context
                                                .read<JobProvider>()
                                                .updateJobStatus(
                                                  job,
                                                  value,
                                                  sourceTag: 'manage_jobs_new',
                                                );
                                            if (!mounted) return;
                                            ScaffoldMessenger.of(context)
                                                .showSnackBar(
                                              SnackBar(
                                                content: Text(
                                                  'Job marked as ${value == 'active' ? 'Active' : 'Closed'}',
                                                ),
                                                backgroundColor: AppTheme.success,
                                              ),
                                            );
                                          } catch (e) {
                                            if (!mounted) return;
                                            ScaffoldMessenger.of(context)
                                                .showSnackBar(
                                              SnackBar(
                                                content:
                                                    Text('Error: ${e.toString()}'),
                                                backgroundColor: AppTheme.error,
                                              ),
                                            );
                                          }
                                        },
                                      ),
                                    ),
                                    const Spacer(),
                                    IconButton(
                                      tooltip: 'Delete job',
                                      icon: const Icon(
                                        Icons.delete_outline,
                                        color: Colors.redAccent,
                                      ),
                                      onPressed: () => _confirmDelete(context, job),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    );
                  },
                  childCount: filteredJobs.length,
                ),
              ),
            ),

          const SliverToBoxAdapter(child: SizedBox(height: 20)),
        ],
      ),
    );
  }

  String _normalizeStatus(String? status) {
    final normalized = (status ?? '').toString().trim().toLowerCase();
    if (normalized.contains('closed') || normalized == 'closed') {
      return 'closed';
    }
    return 'active';
  }

  bool _matchesStatus(dynamic job, String status) {
    final jobStatus = _normalizeStatus(job.status);
    if (status == 'All') return true;
    if (status == 'Active') return jobStatus == 'active';
    if (status == 'Closed') return jobStatus == 'closed';
    return true;
  }

  Future<void> _confirmDelete(BuildContext context, dynamic job) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete job?'),
        content: const Text('This will remove the job and its applications.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) {
      return;
    }
    try {
      await context.read<JobProvider>().deleteJob(
        job,
        sourceTag: 'manage_jobs_new',
      );
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Job deleted')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to delete job: $e')),
      );
    }
  }
}

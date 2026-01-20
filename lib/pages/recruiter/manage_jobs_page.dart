import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../models/job.dart';
import '../../providers/job_provider.dart';

class ManageJobsPage extends StatefulWidget {
  const ManageJobsPage({super.key});

  @override
  State<ManageJobsPage> createState() => _ManageJobsPageState();
}

class _ManageJobsPageState extends State<ManageJobsPage> {
  String _filterStatus = 'All';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      print('[${DateTime.now()}] [manage_jobs_page] addPostFrameCallback fired');
      print('[${DateTime.now()}] [manage_jobs_page] calling loadPostedJobs() from source=manage_jobs');
      context.read<JobProvider>().loadPostedJobs(sourceTag: 'manage_jobs');
    });
  }

  @override
  Widget build(BuildContext context) {
    final jobs = context.watch<JobProvider>();
    final filteredJobs = _filterStatus == 'All'
        ? jobs.postedJobs
        : jobs.postedJobs.where((job) => _matchesStatus(job, _filterStatus)).toList();

    return Column(
      children: [
        // Title
        Padding(
          padding: const EdgeInsets.all(16),
          child: const Row(
            children: [
              Text(
                'Manage Jobs',
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
              children: ['All', 'Active', 'Closed']
                  .map((status) => Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: FilterChip(
                          label: Text(status),
                          selected: _filterStatus == status,
                          onSelected: (selected) =>
                              setState(() => _filterStatus = status),
                        ),
                      ))
                  .toList(),
            ),
          ),
        ),
        const Divider(height: 1),
        // List
        Expanded(
          child: jobs.isLoading
              ? const Center(child: CircularProgressIndicator())
              : filteredJobs.isEmpty
                  ? const Center(
                      child: Text('No jobs'),
                    )
                  : ListView.separated(
                      padding: const EdgeInsets.all(16),
                      itemCount: filteredJobs.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 12),
                      itemBuilder: (context, index) {
                        final job = filteredJobs[index];
                        final statusValue = _normalizeStatus(job.status);
                        return Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(color: const Color(0xFFE2E8F0)),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                job.jobTitle,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                job.location ?? 'Remote',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(color: Colors.black54),
                              ),
                              const SizedBox(height: 12),
                              Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      'Status: ${_labelForStatus(statusValue)}',
                                      maxLines: 1,
                                      overflow: TextOverflow.ellipsis,
                                      style: const TextStyle(fontWeight: FontWeight.w600),
                                    ),
                                  ),
                                  const SizedBox(width: 8),
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
                                          await context.read<JobProvider>().updateJobStatus(
                                                job,
                                                value,
                                                sourceTag: 'manage_jobs',
                                              );
                                          if (!mounted) return;
                                          ScaffoldMessenger.of(context).showSnackBar(
                                            SnackBar(
                                              content: Text('Job marked as ${_labelForStatus(value)}'),
                                            ),
                                          );
                                        } catch (e) {
                                          if (!mounted) return;
                                          ScaffoldMessenger.of(context).showSnackBar(
                                            SnackBar(
                                              content: Text('Failed to update status: $e'),
                                            ),
                                          );
                                        }
                                      },
                                    ),
                                  ),
                                  IconButton(
                                    tooltip: 'Delete job',
                                    icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                                    onPressed: () => _confirmDelete(context, job),
                                  ),
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
  }

  bool _matchesStatus(Job job, String filter) {
    final status = _normalizeStatus(job.status);
    if (filter == 'Active') {
      return status == 'active';
    }
    if (filter == 'Closed') {
      return status == 'closed';
    }
    return true;
  }

  String _normalizeStatus(String? status) {
    final value = (status ?? 'active').toLowerCase();
    return value == 'closed' ? 'closed' : 'active';
  }

  String _labelForStatus(String status) {
    return status == 'closed' ? 'Closed' : 'Active';
  }

  Future<void> _confirmDelete(BuildContext context, Job job) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete job?'),
        content: const Text('This will remove the job and its applications.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
          TextButton(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
        ],
      ),
    );
    if (confirmed != true) {
      return;
    }
    try {
      await context.read<JobProvider>().deleteJob(job, sourceTag: 'manage_jobs');
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

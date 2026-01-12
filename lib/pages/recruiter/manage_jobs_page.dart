import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/job_provider.dart';

class ManageJobsPage extends StatefulWidget {
  const ManageJobsPage({super.key});

  @override
  State<ManageJobsPage> createState() => _ManageJobsPageState();
}

class _ManageJobsPageState extends State<ManageJobsPage> {
  String _filterStatus = 'All';

  @override
  Widget build(BuildContext context) {
    final jobs = context.watch<JobProvider>();
    final filteredJobs = _filterStatus == 'All'
        ? jobs.postedJobs
        : jobs.postedJobs.where((job) => true).toList();

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
          child: filteredJobs.isEmpty
              ? const Center(
                  child: Text('No jobs'),
                )
              : ListView.separated(
                  padding: const EdgeInsets.all(16),
                  itemCount: filteredJobs.length,
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, index) {
                    final job = filteredJobs[index];
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
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            job.location ?? 'Remote',
                            style: const TextStyle(color: Colors.black54),
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
}

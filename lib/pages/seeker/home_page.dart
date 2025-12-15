import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/job.dart';
import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../widgets/job_card.dart';
import 'job_details_page.dart';

class SeekerHomePage extends StatefulWidget {
  const SeekerHomePage({super.key});

  @override
  State<SeekerHomePage> createState() => _SeekerHomePageState();
}

class _SeekerHomePageState extends State<SeekerHomePage> {
  final searchCtrl = TextEditingController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final jobs = context.read<JobProvider>();
      final auth = context.read<AuthProvider>();
      jobs.loadJobs();
      if (auth.session != null) {
        jobs.refreshRecommendations(auth.session!.profile.id);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final jobs = context.watch<JobProvider>();
    final name = auth.session?.profile.fullName.split(' ').first ?? 'Explorer';

    return RefreshIndicator(
      onRefresh: () async {
        await jobs.loadJobs();
        if (auth.session != null) {
          await jobs.refreshRecommendations(auth.session!.profile.id);
        }
      },
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Hello, $name 👋', style: const TextStyle(color: Colors.black54, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  const Text('Find your next role', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
                ],
              ),
              const CircleAvatar(
                backgroundColor: Color(0xFF4F46E5),
                child: Icon(Icons.person, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 16),
          TextField(
            controller: searchCtrl,
            decoration: const InputDecoration(
              hintText: 'Search for jobs, companies...',
              prefixIcon: Icon(Icons.search_rounded),
              suffixIcon: Icon(Icons.tune_rounded, color: Color(0xFF4F46E5)),
            ),
          ),
          const SizedBox(height: 14),
          SizedBox(
            height: 44,
            child: ListView(
              scrollDirection: Axis.horizontal,
              children: const [
                _FilterChip(label: 'All Jobs', selected: true),
                _FilterChip(label: 'Engineering'),
                _FilterChip(label: 'Design'),
                _FilterChip(label: 'Marketing'),
                _FilterChip(label: 'Business'),
              ],
            ),
          ),
          const SizedBox(height: 20),
          _sectionHeader('Recommended for you'),
          const SizedBox(height: 10),
          if (jobs.recommendations.isEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFFEEF2FF),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Text('Recommendations will appear here once you start exploring jobs.'),
            )
          else
            Column(
              children: jobs.recommendations.map((rec) {
                final job = _findJob(jobs.jobs, rec.jobId);
                if (job == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: JobCard(
                    job: job,
                    trailing: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        const Text('Hybrid score', style: TextStyle(color: Colors.black54, fontSize: 12)),
                        Text(rec.finalScore.toStringAsFixed(2), style: const TextStyle(fontWeight: FontWeight.w800)),
                      ],
                    ),
                    onTap: () => _openDetails(context, job),
                  ),
                );
              }).toList(),
            ),
          const SizedBox(height: 16),
          _sectionHeader('Latest openings'),
          const SizedBox(height: 10),
          if (jobs.isLoading)
            const Center(child: Padding(padding: EdgeInsets.all(20), child: CircularProgressIndicator()))
          else
            ...jobs.jobs.map((job) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: JobCard(job: job, onTap: () => _openDetails(context, job)),
                ))
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
        TextButton(onPressed: () {}, child: const Text('See all')),
      ],
    );
  }

  Job? _findJob(List<Job> jobs, String jobId) {
    try {
      return jobs.firstWhere((j) => j.jobId == jobId);
    } catch (_) {
      return null;
    }
  }

  void _openDetails(BuildContext context, Job job) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => JobDetailsPage(job: job)));
  }
}

class _FilterChip extends StatelessWidget {
  const _FilterChip({required this.label, this.selected = false});

  final String label;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ChoiceChip(
        label: Text(label),
        selected: selected,
        onSelected: (_) {},
      ),
    );
  }
}

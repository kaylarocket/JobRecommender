import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/job.dart';
import '../../models/recommendation.dart';
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
  Timer? _debounce;
  String? _selectedCategory;

  @override
  void initState() {
    super.initState();
    searchCtrl.addListener(_onSearchChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      print('[${DateTime.now()}] [job_seeker_home] addPostFrameCallback fired');
      print('[${DateTime.now()}] [job_seeker_home] calling loadJobs() and refreshRecommendations() from source=job_seeker_home');
      final jobs = context.read<JobProvider>();
      final auth = context.read<AuthProvider>();
      jobs.loadJobs(sourceTag: 'job_seeker_home');
      if (auth.session != null) {
        jobs.refreshRecommendations(auth.session!.profile.id, sourceTag: 'job_seeker_home');
      }
    });
  }

  @override
  void dispose() {
    searchCtrl.removeListener(_onSearchChanged);
    searchCtrl.dispose();
    _debounce?.cancel();
    super.dispose();
  }

  void _onSearchChanged() {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 500), () {
      _performSearch();
    });
  }

  void _performSearch() {
    print('[${DateTime.now()}] [job_seeker_home] _performSearch() called: query="${searchCtrl.text}", category=$_selectedCategory');
    final jobs = context.read<JobProvider>();
    jobs.loadJobs(
      query: searchCtrl.text.isEmpty ? null : searchCtrl.text,
      category: _selectedCategory,
      sourceTag: 'job_seeker_home_search',
    );
  }

  void _onCategorySelected(String? category) {
    setState(() {
      _selectedCategory = category == 'All Jobs' ? null : category;
    });
    _performSearch();
  }

  void _clearSearch() {
    searchCtrl.clear();
    setState(() {
      _selectedCategory = null;
    });
    context.read<JobProvider>().clearSearch();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    final jobs = context.watch<JobProvider>();
    final name = auth.session?.profile.fullName.split(' ').first ?? 'Explorer';

    return RefreshIndicator(
      onRefresh: () async {
        print('[${DateTime.now()}] [job_seeker_home] onRefresh triggered');
        await jobs.loadJobs(sourceTag: 'job_seeker_home_refresh');
        if (auth.session != null) {
          await jobs.refreshRecommendations(auth.session!.profile.id, sourceTag: 'job_seeker_home_refresh');
        }
      },
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 20, 16, 16),
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Hello, $name 👋',
                      style: const TextStyle(
                          color: Colors.black54, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  const Text('Find your next role',
                      style:
                          TextStyle(fontSize: 22, fontWeight: FontWeight.w800)),
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
            decoration: InputDecoration(
              hintText: 'Search for jobs, companies...',
              prefixIcon: const Icon(Icons.search_rounded),
              suffixIcon: searchCtrl.text.isNotEmpty || _selectedCategory != null
                  ? IconButton(
                      icon: const Icon(Icons.clear, color: Colors.grey),
                      onPressed: _clearSearch,
                    )
                  : const Icon(Icons.tune_rounded, color: Color(0xFF4F46E5)),
            ),
          ),
          const SizedBox(height: 14),
          SizedBox(
            height: 44,
            child: ListView(
              scrollDirection: Axis.horizontal,
              children: [
                _FilterChip(
                  label: 'All Jobs',
                  selected: _selectedCategory == null,
                  onSelected: () => _onCategorySelected(null),
                ),
                _FilterChip(
                  label: 'Engineering',
                  selected: _selectedCategory == 'Engineering',
                  onSelected: () => _onCategorySelected('Engineering'),
                ),
                _FilterChip(
                  label: 'Design',
                  selected: _selectedCategory == 'Design',
                  onSelected: () => _onCategorySelected('Design'),
                ),
                _FilterChip(
                  label: 'Marketing',
                  selected: _selectedCategory == 'Marketing',
                  onSelected: () => _onCategorySelected('Marketing'),
                ),
                _FilterChip(
                  label: 'Business',
                  selected: _selectedCategory == 'Business',
                  onSelected: () => _onCategorySelected('Business'),
                ),
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
              child: const Text(
                  'Recommendations will appear here once you start exploring jobs.'),
            )
          else
            Column(
              children: jobs.recommendations.map((rec) {
                final job = _findJob(jobs.jobs, rec.jobId) ?? _jobFromRecommendation(rec);
                if (job == null) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: JobCard(
                    job: job,
                    trailing: Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        const Text('Hybrid score',
                            style:
                                TextStyle(color: Colors.black54, fontSize: 12)),
                        Text(rec.finalScore.toStringAsFixed(2),
                            style:
                                const TextStyle(fontWeight: FontWeight.w800)),
                      ],
                    ),
                    onTap: () => _openDetails(context, job),
                  ),
                );
              }).toList(),
            ),
          const SizedBox(height: 16),
          _sectionHeader(
            jobs.currentSearchQuery != null || jobs.currentCategory != null
                ? 'Search Results'
                : 'Latest openings',
          ),
          const SizedBox(height: 10),
          if (jobs.isLoading)
            const Center(
                child: Padding(
                    padding: EdgeInsets.all(20),
                    child: CircularProgressIndicator()))
          else if (jobs.jobs.isEmpty)
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: const Color(0xFFF3F4F6),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                children: [
                  const Icon(Icons.search_off, size: 48, color: Colors.black38),
                  const SizedBox(height: 12),
                  const Text(
                    'No jobs found',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    jobs.currentSearchQuery != null
                        ? 'Try adjusting your search terms'
                        : 'Check back later for new opportunities',
                    style: const TextStyle(color: Colors.black54),
                  ),
                ],
              ),
            )
          else
            ...jobs.jobs.map((job) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: JobCard(
                      job: job, onTap: () => _openDetails(context, job)),
                ))
        ],
      ),
    );
  }

  Widget _sectionHeader(String title) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(title,
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
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

  Job? _jobFromRecommendation(Recommendation rec) {
    if (rec.jobId.isEmpty) return null;
    final title = (rec.jobTitle ?? '').trim();
    return Job(
      jobId: rec.jobId,
      jobTitle: title.isEmpty ? 'Recommended role' : title,
      company: rec.company,
      location: rec.location,
      category: rec.category,
      salary: rec.salary,
      descriptions: null,
    );
  }

  void _openDetails(BuildContext context, Job job) {
    Navigator.push(
        context, MaterialPageRoute(builder: (_) => JobDetailsPage(job: job)));
  }
}

class _FilterChip extends StatelessWidget {
  const _FilterChip({
    required this.label,
    this.selected = false,
    required this.onSelected,
  });

  final String label;
  final bool selected;
  final VoidCallback onSelected;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: ChoiceChip(
        label: Text(label),
        selected: selected,
        onSelected: (_) => onSelected(),
      ),
    );
  }
}

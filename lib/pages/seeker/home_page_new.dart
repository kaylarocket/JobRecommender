import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/modern_widgets.dart';
import 'job_details_page_new.dart';

/// Modern redesigned home page with clean card layout and improved UX
class SeekerHomePageNew extends StatefulWidget {
  const SeekerHomePageNew({super.key});

  @override
  State<SeekerHomePageNew> createState() => _SeekerHomePageNewState();
}

class _SeekerHomePageNewState extends State<SeekerHomePageNew> {
  final searchCtrl = TextEditingController();
  Timer? _debounce;
  String? _selectedCategory;
  bool _showFilters = false;

  final List<String> categories = [
    'All',
    'Engineering',
    'Design',
    'Product',
    'Marketing',
    'Sales',
    'HR',
  ];

  @override
  void initState() {
    super.initState();
    searchCtrl.addListener(_onSearchChanged);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final jobs = context.read<JobProvider>();
      final auth = context.read<AuthProvider>();
      jobs.loadJobs(sourceTag: 'seeker_home_new');
      if (auth.session != null) {
        jobs.refreshRecommendations(auth.session!.profile.id,
            sourceTag: 'seeker_home_new');
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
    final jobs = context.read<JobProvider>();
    jobs.loadJobs(
      query: searchCtrl.text.isEmpty ? null : searchCtrl.text,
      category: _selectedCategory == 'All' || _selectedCategory == null
          ? null
          : _selectedCategory,
      sourceTag: 'seeker_home_search',
    );
  }

  void _onCategorySelected(String? category) {
    setState(() {
      _selectedCategory = category == 'All' ? null : category;
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
        await jobs.loadJobs(sourceTag: 'seeker_home_refresh');
        if (auth.session != null) {
          await jobs.refreshRecommendations(auth.session!.profile.id,
              sourceTag: 'seeker_home_refresh');
        }
      },
      color: AppTheme.primary,
      child: CustomScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        slivers: [
          // Header with greeting
          SliverAppBar(
            backgroundColor: AppTheme.surface,
            elevation: 0,
            scrolledUnderElevation: 0,
            pinned: true,
            expandedHeight: 120,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                color: AppTheme.surface,
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.end,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              "Hello, $name 👋",
                              style: Theme.of(context)
                                  .textTheme
                                  .bodyMedium
                                  ?.copyWith(
                                    fontWeight: FontWeight.w600,
                                    color: AppTheme.mutedText,
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              "Let's find your dream job",
                              style: Theme.of(context)
                                  .textTheme
                                  .headlineSmall
                                  ?.copyWith(
                                    fontWeight: FontWeight.w800,
                                    fontSize: 24,
                                  ),
                            ),
                          ],
                        ),
                        Container(
                          width: 40,
                          height: 40,
                          decoration: BoxDecoration(
                            color: AppTheme.primary.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Icon(
                            Icons.person_outline_rounded,
                            color: AppTheme.primary,
                            size: 20,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                  ],
                ),
              ),
            ),
          ),
          // Search and filters
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
              child: Column(
                children: [
                  ModernSearchBar(
                    controller: searchCtrl,
                    onChanged: (_) {},
                    onFilterTap: () {
                      setState(() => _showFilters = !_showFilters);
                    },
                    onClear: _clearSearch,
                  ),
                  const SizedBox(height: 16),
                  if (_showFilters || _selectedCategory != null)
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Categories',
                          style: Theme.of(context).textTheme.titleSmall,
                        ),
                        const SizedBox(height: 8),
                        FilterChipsGroup(
                          chips: categories,
                          selectedChip: _selectedCategory ?? 'All',
                          onChipSelected: _onCategorySelected,
                        ),
                        const SizedBox(height: 16),
                      ],
                    ),
                ],
              ),
            ),
          ),
          // Recommended For You section header
          if (jobs.recommendations.isNotEmpty)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
                child: Row(
                  children: [
                    Container(
                      width: 4,
                      height: 24,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [AppTheme.primary, AppTheme.accent],
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                        ),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'Recommended For You',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      '${jobs.recommendations.length}',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.primary,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          // Loading state
          if (jobs.isLoading)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  children: List.generate(3, (_) => const LoadingJobCard()),
                ),
              ),
            )
          // Recommendations section
          else if (jobs.recommendations.isNotEmpty)
            SliverList(
              delegate: SliverChildBuilderDelegate(
                (context, index) {
                  final rec = jobs.recommendations[index];
                  final isSaved = jobs.isJobSaved(rec.jobId);

                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: ModernJobCard(
                      title: rec.jobTitle ?? 'Job',
                      company: rec.company ?? 'Company',
                      location: rec.location ?? 'Remote',
                      category: rec.category ?? 'General',
                      salary: rec.salary ?? 'Not disclosed',
                      isSaved: isSaved,
                      applicantCount: null,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => JobDetailsPageNew(job: rec.toJob()),
                          ),
                        );
                      },
                      onSaveToggle: () async {
                        await jobs.toggleSavedJob(rec.toJob());
                        setState(() {});
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                jobs.isJobSaved(rec.jobId)
                                    ? 'Saved for later'
                                    : 'Removed from saved',
                              ),
                              duration: const Duration(milliseconds: 800),
                            ),
                          );
                        }
                      },
                    ),
                  );
                },
                childCount: jobs.recommendations.length,
              ),
            ),
          // Latest Jobs section header
          if (!jobs.isLoading && jobs.jobs.isNotEmpty)
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 24, 16, 12),
                child: Row(
                  children: [
                    Container(
                      width: 4,
                      height: 24,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: [AppTheme.accent, AppTheme.primary],
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                        ),
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      'Latest Job Openings',
                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const Spacer(),
                    Text(
                      '${jobs.jobs.length}',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.accent,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          // No results
          if (!jobs.isLoading && jobs.jobs.isEmpty && jobs.recommendations.isEmpty)
            SliverToBoxAdapter(
              child: EmptyState(
                icon: Icons.search_off_rounded,
                title: 'No jobs found',
                subtitle: 'Try adjusting your search or filters',
                actionLabel: 'Clear filters',
                onAction: _clearSearch,
              ),
            )
          // Latest jobs list
          else if (!jobs.isLoading && jobs.jobs.isNotEmpty)
          // Latest jobs list (continued)
            SliverList(
              delegate: SliverChildBuilderDelegate(
                (context, index) {
                  final job = jobs.jobs[index];
                  final isSaved = jobs.isJobSaved(job.jobId);

                  return Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: ModernJobCard(
                      title: job.jobTitle,
                      company: job.company ?? 'Company',
                      location: job.location ?? 'Remote',
                      category: job.category ?? 'General',
                      salary: job.salary ?? 'Not disclosed',
                      isSaved: isSaved,
                      applicantCount: null,
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => JobDetailsPageNew(job: job),
                          ),
                        );
                      },
                      onSaveToggle: () async {
                        await jobs.toggleSavedJob(job);
                        setState(() {});
                        if (context.mounted) {
                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                jobs.isJobSaved(job.jobId)
                                    ? 'Saved for later'
                                    : 'Removed from saved',
                              ),
                              duration: const Duration(milliseconds: 800),
                            ),
                          );
                        }
                      },
                    ),
                  );
                },
                childCount: jobs.jobs.length,
              ),
            ),
          // Bottom padding
          SliverToBoxAdapter(
            child: SizedBox(height: MediaQuery.of(context).padding.bottom + 16),
          ),
        ],
      ),
    );
  }
}

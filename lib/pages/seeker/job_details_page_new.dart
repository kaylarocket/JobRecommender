import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../models/job.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/modern_widgets.dart';

/// Modern job details page with hero section and clean layout
class JobDetailsPageNew extends StatefulWidget {
  const JobDetailsPageNew({super.key, required this.job});

  final Job job;

  @override
  State<JobDetailsPageNew> createState() => _JobDetailsPageNewState();
}

class _JobDetailsPageNewState extends State<JobDetailsPageNew> {
  bool _isApplied = false;

  @override
  Widget build(BuildContext context) {
    final jobProvider = context.watch<JobProvider>();
    final isSaved = jobProvider.isJobSaved(widget.job.jobId);

    return Scaffold(
      body: CustomScrollView(
        slivers: [
          // App bar with hero image
          SliverAppBar(
            expandedHeight: 200,
            pinned: true,
            elevation: 0,
            scrolledUnderElevation: 0,
            backgroundColor: AppTheme.surface,
            flexibleSpace: FlexibleSpaceBar(
              background: Container(
                decoration: BoxDecoration(
                  gradient: AppTheme.gradientBackground,
                ),
                child: Stack(
                  children: [
                    // Decorative circles
                    Positioned(
                      top: -50,
                      right: -50,
                      child: Container(
                        width: 200,
                        height: 200,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: AppTheme.primary.withOpacity(0.1),
                        ),
                      ),
                    ),
                    // Company info
                    Padding(
                      padding: const EdgeInsets.fromLTRB(16, 16, 16, 56),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.end,
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              color: AppTheme.primary,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Center(
                              child: Text(
                                (widget.job.company?.isNotEmpty ?? false)
                                    ? widget.job.company![0].toUpperCase()
                                    : '?',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w800,
                                  fontSize: 24,
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            widget.job.company ?? 'Company',
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
                  ],
                ),
              ),
              titlePadding: const EdgeInsets.all(16),
              title: Text(
                widget.job.jobTitle,
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: IconButton(
                  icon: Icon(
                    isSaved ? Icons.bookmark_rounded : Icons.bookmark_border_rounded,
                    color: isSaved ? AppTheme.primary : Colors.grey,
                    size: 24,
                  ),
                  onPressed: () async {
                    await jobProvider.toggleSavedJob(widget.job);
                    setState(() {});
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text(
                          jobProvider.isJobSaved(widget.job.jobId)
                              ? 'Saved for later'
                              : 'Removed from saved',
                        ),
                        duration: const Duration(milliseconds: 800),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
          // Content
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 24, 16, 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Quick info chips
                  Wrap(
                    spacing: 12,
                    runSpacing: 12,
                    children: [
                      _DetailChip(
                        icon: Icons.location_on_outlined,
                        label: widget.job.location ?? 'Remote',
                      ),
                      _DetailChip(
                        icon: Icons.category_outlined,
                        label: widget.job.category ?? 'General',
                      ),
                      _DetailChip(
                        icon: Icons.attach_money_rounded,
                        label: widget.job.salary ?? 'Not disclosed',
                        isPrimary: true,
                      ),
                    ],
                  ),
                  const SizedBox(height: 28),
                  // Description section
                  if (widget.job.descriptions != null &&
                      widget.job.descriptions!.isNotEmpty)
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'About this job',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: AppTheme.muted,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            widget.job.descriptions!,
                            style: Theme.of(context)
                                .textTheme
                                .bodyMedium
                                ?.copyWith(
                                  height: 1.6,
                                  color: AppTheme.mutedText,
                                ),
                          ),
                        ),
                        const SizedBox(height: 28),
                      ],
                    ),
                  // Requirements section (if available)
                  _SectionTitle(title: 'Key Information'),
                  const SizedBox(height: 12),
                  _InfoRow(
                    icon: Icons.business_outlined,
                    title: 'Company',
                    value: widget.job.company ?? 'Not specified',
                  ),
                  _InfoRow(
                    icon: Icons.location_on_outlined,
                    title: 'Location',
                    value: widget.job.location ?? 'Remote',
                  ),
                  if (widget.job.category != null)
                    _InfoRow(
                      icon: Icons.category_outlined,
                      title: 'Category',
                      value: widget.job.category!,
                    ),
                  if (widget.job.salary != null)
                    _InfoRow(
                      icon: Icons.attach_money_rounded,
                      title: 'Salary',
                      value: widget.job.salary!,
                    ),
                  const SizedBox(height: 32),
                ],
              ),
            ),
          ),
        ],
      ),
      // Sticky action buttons
      bottomNavigationBar: Container(
        padding: EdgeInsets.fromLTRB(
          16,
          12,
          16,
          16 + MediaQuery.of(context).padding.bottom,
        ),
        decoration: BoxDecoration(
          color: AppTheme.surface,
          border: Border(
            top: BorderSide(color: AppTheme.divider, width: 1),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 16,
              offset: const Offset(0, -4),
            ),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () async {
                      await jobProvider.toggleSavedJob(widget.job);
                      setState(() {});
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text(
                            jobProvider.isJobSaved(widget.job.jobId)
                                ? 'Saved for later'
                                : 'Removed from saved',
                          ),
                          duration: const Duration(milliseconds: 800),
                        ),
                      );
                    },
                    icon: Icon(
                      isSaved ? Icons.bookmark_rounded : Icons.bookmark_border_rounded,
                    ),
                    label: Text(isSaved ? 'Saved' : 'Save'),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  flex: 1,
                  child: ElevatedButton.icon(
                    onPressed: _isApplied
                        ? null
                        : () async {
                            await jobProvider.apply(widget.job);
                            setState(() => _isApplied = true);
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Application submitted 🎉'),
                                duration: Duration(seconds: 2),
                              ),
                            );
                            Future.delayed(const Duration(seconds: 2), () {
                              Navigator.pop(context);
                            });
                          },
                    icon: Icon(
                      _isApplied ? Icons.check_circle : Icons.send_rounded,
                    ),
                    label: Text(_isApplied ? 'Applied' : 'Apply Now'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// Detail chip for job info
class _DetailChip extends StatelessWidget {
  const _DetailChip({
    required this.icon,
    required this.label,
    this.isPrimary = false,
  });

  final IconData icon;
  final String label;
  final bool isPrimary;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: isPrimary
            ? AppTheme.primary.withOpacity(0.1)
            : AppTheme.muted,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isPrimary ? AppTheme.primary.withOpacity(0.3) : Colors.transparent,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 16,
            color: isPrimary ? AppTheme.primary : AppTheme.mutedText,
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: isPrimary ? AppTheme.primary : AppTheme.mutedText,
                  fontWeight: FontWeight.w600,
                ),
          ),
        ],
      ),
    );
  }
}

/// Section title widget
class _SectionTitle extends StatelessWidget {
  const _SectionTitle({required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    return Text(
      title,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w700,
          ),
    );
  }
}

/// Information row for job details
class _InfoRow extends StatelessWidget {
  const _InfoRow({
    required this.icon,
    required this.title,
    required this.value,
  });

  final IconData icon;
  final String title;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.primary.withOpacity(0.08),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(
              icon,
              size: 18,
              color: AppTheme.primary,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: AppTheme.mutedText,
                      ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

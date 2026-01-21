import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Modern job card with gradient accent and clean typography
class ModernJobCard extends StatefulWidget {
  const ModernJobCard({
    super.key,
    required this.title,
    required this.company,
    required this.location,
    required this.category,
    required this.salary,
    required this.isSaved,
    required this.onTap,
    required this.onSaveToggle,
    this.imageUrl,
    this.applicantCount,
  });

  final String title;
  final String company;
  final String location;
  final String category;
  final String salary;
  final bool isSaved;
  final VoidCallback onTap;
  final VoidCallback onSaveToggle;
  final String? imageUrl;
  final int? applicantCount;

  @override
  State<ModernJobCard> createState() => _ModernJobCardState();
}

class _ModernJobCardState extends State<ModernJobCard>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => _controller.forward(),
      onTapUp: (_) {
        _controller.reverse();
        widget.onTap();
      },
      onTapCancel: () => _controller.reverse(),
      child: ScaleTransition(
        scale: Tween<double>(begin: 1.0, end: 0.98).animate(
          CurvedAnimation(parent: _controller, curve: Curves.easeInOut),
        ),
        child: Container(
          margin: const EdgeInsets.only(bottom: 12),
          decoration: BoxDecoration(
            color: AppTheme.surface,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppTheme.divider, width: 1),
            boxShadow: AppTheme.boxShadowSmall,
          ),
          child: Column(
            children: [
              // Header with gradient accent
              Container(
                height: 4,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      AppTheme.primary,
                      AppTheme.accent,
                    ],
                  ),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    topRight: Radius.circular(16),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Title and save button
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                widget.title,
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style:
                                    Theme.of(context).textTheme.titleMedium,
                              ),
                              const SizedBox(height: 4),
                              Text(
                                widget.company,
                                style: Theme.of(context)
                                    .textTheme
                                    .bodySmall
                                    ?.copyWith(
                                      color: AppTheme.mutedText,
                                      fontWeight: FontWeight.w500,
                                    ),
                              ),
                            ],
                          ),
                        ),
                        IconButton(
                          onPressed: widget.onSaveToggle,
                          icon: Icon(
                            widget.isSaved
                                ? Icons.bookmark_rounded
                                : Icons.bookmark_border_rounded,
                            color:
                                widget.isSaved ? AppTheme.primary : Colors.grey,
                          ),
                          tooltip: widget.isSaved
                              ? 'Remove from saved'
                              : 'Save for later',
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    // Location, category, and salary chips
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _CompactChip(
                          icon: Icons.location_on_outlined,
                          label: widget.location,
                        ),
                        _CompactChip(
                          icon: Icons.category_outlined,
                          label: widget.category,
                        ),
                        if (widget.salary.isNotEmpty)
                          _CompactChip(
                            icon: Icons.attach_money_rounded,
                            label: widget.salary,
                            isPrimary: true,
                          ),
                      ],
                    ),
                    if (widget.applicantCount != null) ...[
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Icon(
                            Icons.people_alt_outlined,
                            size: 14,
                            color: AppTheme.mutedText,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            '${widget.applicantCount} applicants',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Compact chip for job card details
class _CompactChip extends StatelessWidget {
  const _CompactChip({
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
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: isPrimary
            ? AppTheme.primary.withOpacity(0.08)
            : AppTheme.muted,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 14,
            color: isPrimary ? AppTheme.primary : AppTheme.mutedText,
          ),
          const SizedBox(width: 4),
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

/// Modern search bar with filter icon
class ModernSearchBar extends StatefulWidget {
  const ModernSearchBar({
    super.key,
    required this.controller,
    required this.onChanged,
    required this.onFilterTap,
    this.hintText = 'Search jobs, companies...',
    this.onClear,
  });

  final TextEditingController controller;
  final ValueChanged<String> onChanged;
  final VoidCallback onFilterTap;
  final String hintText;
  final VoidCallback? onClear;

  @override
  State<ModernSearchBar> createState() => _ModernSearchBarState();
}

class _ModernSearchBarState extends State<ModernSearchBar> {
  late FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _focusNode = FocusNode();
  }

  @override
  void dispose() {
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        boxShadow: AppTheme.boxShadowSmall,
      ),
      child: TextField(
        controller: widget.controller,
        focusNode: _focusNode,
        onChanged: widget.onChanged,
        decoration: InputDecoration(
          hintText: widget.hintText,
          hintStyle: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: AppTheme.mutedText,
              ),
          prefixIcon: const Icon(Icons.search_rounded, size: 20),
          suffixIcon: widget.controller.text.isNotEmpty
              ? IconButton(
                  onPressed: () {
                    widget.controller.clear();
                    widget.onChanged('');
                    widget.onClear?.call();
                  },
                  icon: const Icon(Icons.close_rounded, size: 20),
                )
              : IconButton(
                  onPressed: widget.onFilterTap,
                  icon: const Icon(Icons.tune_rounded, size: 20),
                ),
          filled: true,
          fillColor: AppTheme.surface,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: AppTheme.divider),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: AppTheme.divider),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
            borderSide: const BorderSide(color: AppTheme.primary, width: 2),
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 12,
          ),
        ),
      ),
    );
  }
}

/// Horizontal scrollable filter chips
class FilterChipsGroup extends StatefulWidget {
  const FilterChipsGroup({
    super.key,
    required this.chips,
    required this.selectedChip,
    required this.onChipSelected,
  });

  final List<String> chips;
  final String? selectedChip;
  final Function(String?) onChipSelected;

  @override
  State<FilterChipsGroup> createState() => _FilterChipsGroupState();
}

class _FilterChipsGroupState extends State<FilterChipsGroup> {
  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 40,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: widget.chips.length,
        itemBuilder: (context, index) {
          final chip = widget.chips[index];
          final isSelected = widget.selectedChip == chip;

          return Padding(
            padding: EdgeInsets.only(right: index == widget.chips.length - 1 ? 0 : 8),
            child: FilterChip(
              selected: isSelected,
              onSelected: (selected) {
                widget.onChipSelected(selected ? chip : null);
              },
              label: Text(chip),
              backgroundColor: AppTheme.muted,
              selectedColor: AppTheme.primary.withOpacity(0.15),
              side: BorderSide(
                color: isSelected ? AppTheme.primary : Colors.transparent,
                width: isSelected ? 2 : 0,
              ),
              labelStyle: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: isSelected ? AppTheme.primary : AppTheme.mutedText,
                  ),
            ),
          );
        },
      ),
    );
  }
}

/// Empty state widget for no results
class EmptyState extends StatelessWidget {
  const EmptyState({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    this.actionLabel,
    this.onAction,
  });

  final IconData icon;
  final String title;
  final String subtitle;
  final String? actionLabel;
  final VoidCallback? onAction;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 64),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Icon(
                icon,
                size: 56,
                color: AppTheme.primary,
              ),
            ),
            const SizedBox(height: 20),
            Text(
              title,
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              subtitle,
              style: Theme.of(context).textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            if (actionLabel != null && onAction != null) ...[
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: onAction,
                child: Text(actionLabel!),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Loading state with shimmer animation
class LoadingJobCard extends StatelessWidget {
  const LoadingJobCard({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.divider, width: 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _ShimmerPlaceholder(height: 16, width: 200),
          const SizedBox(height: 8),
          _ShimmerPlaceholder(height: 14, width: 150),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _ShimmerPlaceholder(height: 24, width: 100),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ShimmerPlaceholder(height: 24, width: 100),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Shimmer placeholder for loading states
class _ShimmerPlaceholder extends StatefulWidget {
  const _ShimmerPlaceholder({
    required this.height,
    required this.width,
  });

  final double height;
  final double width;

  @override
  State<_ShimmerPlaceholder> createState() => _ShimmerPlaceholderState();
}

class _ShimmerPlaceholderState extends State<_ShimmerPlaceholder>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.width,
      height: widget.height,
      child: AnimatedBuilder(
        animation: _controller,
        builder: (context, child) {
          return Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  AppTheme.muted,
                  AppTheme.muted.withOpacity(0.5),
                  AppTheme.muted,
                ],
                stops: [
                  0,
                  _controller.value,
                  1,
                ],
              ),
              borderRadius: BorderRadius.circular(8),
            ),
          );
        },
      ),
    );
  }
}

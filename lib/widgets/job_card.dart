import 'package:flutter/material.dart';

import '../models/job.dart';
import '../theme/app_theme.dart';

class JobCard extends StatelessWidget {
  const JobCard({super.key, required this.job, this.onTap, this.trailing});

  final Job job;
  final VoidCallback? onTap;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final title = _cleanText(job.jobTitle) ?? 'Untitled role';
    final company = _cleanText(job.company);
    final location = _cleanText(job.location);
    final category = _cleanText(job.category);
    final salary = _cleanText(job.salary);
    final description = _cleanPreviewText(job.descriptions);

    // Preview priority: salary first; otherwise description.
    final previewText = salary ?? description;

    // If caller supplies trailing (e.g. bookmark), we don't show the chevron.
    final showChevron = trailing == null;

    final hasMeta = location != null || category != null;

    // Build the text/meta blocks shown under the title area
    final blocks = <Widget>[];
    void addBlock(Widget child) {
      if (blocks.isNotEmpty) blocks.add(const SizedBox(height: 8));
      blocks.add(child);
    }

    // Title
    addBlock(
      Text(
        title,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16),
      ),
    );

    // Company
    if (company != null) {
      addBlock(
        Text(
          company,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            color: Colors.black54,
            fontWeight: FontWeight.w600,
          ),
        ),
      );
    }

    // Chips (location/category)
    if (hasMeta) {
      addBlock(_metaSection(location: location, category: category));
    }

    // Preview row (salary pill OR icon+description) + chevron
    if (previewText != null || showChevron) {
      addBlock(
        _previewRow(
          previewText: previewText,
          isSalary: salary != null,
          showChevron: showChevron,
        ),
      );
    }

    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(20),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0xFFFEFEFF),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: const Color(0xFFE2E8F0)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Left icon block (like your reference)
            Container(
              height: 44,
              width: 44,
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.08),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.work_outline_rounded,
                color: AppTheme.primary,
              ),
            ),
            const SizedBox(width: 12),

            // Main content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: blocks,
              ),
            ),

            // Optional trailing widget (bookmark etc.)
            if (trailing != null)
              Padding(
                padding: const EdgeInsets.only(left: 8),
                child: trailing!,
              ),
          ],
        ),
      ),
    );
  }

  /// Meta chips (keep bordered pills; allow wrapping)
  Widget _metaSection({String? location, String? category}) {
    final chips = <Widget>[];
    if (location != null) chips.add(_pill(Icons.place_outlined, location));
    if (category != null) chips.add(_pill(Icons.category_outlined, category));

    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: chips,
    );
  }

  /// Preview row:
  /// - if salary exists: show salary pill
  /// - else: show icon + description (inline with icon)
  /// - chevron on the right if no custom trailing widget
  Widget _previewRow({
    required String? previewText,
    required bool isSalary,
    required bool showChevron,
  }) {
    Widget left;

    if (previewText == null) {
      left = const SizedBox.shrink();
    } else if (isSalary) {
      left = _salaryPill(previewText);
    } else {
      left = Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.description_outlined, size: 16, color: Colors.black54),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              previewText,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.black87,
                fontWeight: FontWeight.w600,
                height: 1.25,
              ),
            ),
          ),
        ],
      );
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(child: left),
        if (showChevron) ...[
          const SizedBox(width: 8),
          const Padding(
            padding: EdgeInsets.only(top: 2),
            child: Icon(Icons.arrow_forward_ios_rounded, size: 16, color: Colors.black54),
          ),
        ],
      ],
    );
  }

  /// Salary pill (keep as you had; looks good)
  Widget _salaryPill(String salary) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: AppTheme.primary.withOpacity(0.12),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        salary,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(color: AppTheme.primary, fontWeight: FontWeight.w700),
      ),
    );
  }

  /// Bordered chip pill with icon + text (your desired look)
  Widget _pill(IconData icon, String label) {
    return IntrinsicWidth(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: const Color(0xFFF8FAFC),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: const Color(0xFFE2E8F0)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: Colors.black54),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                softWrap: false,
                style: const TextStyle(fontWeight: FontWeight.w600, color: Colors.black87),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ----------------------------
  // Text cleaning helpers
  // ----------------------------

  String? _cleanText(String? value) {
    if (value == null) return null;
    final trimmed = value.trim();
    if (trimmed.isEmpty) return null;
    final lower = trimmed.toLowerCase();
    if (lower == 'nan' || lower == 'null' || lower == 'n/a') return null;
    return trimmed;
  }

  String? _cleanPreviewText(String? value) {
    final cleaned = _cleanText(value);
    if (cleaned == null) return null;

    final lines = cleaned.split(RegExp(r'[\r\n]+'));
    final kept = <String>[];

    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;

      final stripped = _stripLabelPrefix(trimmed);
      final normalized = _cleanText(stripped);
      if (normalized == null) continue;

      kept.add(normalized);
    }

    if (kept.isEmpty) return null;

    final joined = kept.join(' ').replaceAll(RegExp(r'\s+'), ' ').trim();
    return _cleanText(joined);
  }

  String _stripLabelPrefix(String line) {
    final labelPatterns = [
      RegExp(r'^(job\s+)?description\s*:?', caseSensitive: false),
      RegExp(r'^about\s+us\s*:?', caseSensitive: false),
      RegExp(r'^(roles?\s+and\s+responsibilities|responsibilities)\s*:?', caseSensitive: false),
      RegExp(r'^company\s+overview\s*:?', caseSensitive: false),
      RegExp(r'^job\s+summary\s*:?', caseSensitive: false),
    ];

    for (final pattern in labelPatterns) {
      final match = pattern.firstMatch(line);
      if (match != null && match.start == 0) {
        return line.substring(match.end).trim();
      }
    }
    return line;
  }
}

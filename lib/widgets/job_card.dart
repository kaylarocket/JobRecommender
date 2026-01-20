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
    final previewText = salary ?? description;
    final showChevron = trailing == null;
    final hasMeta = location != null || category != null;

    final blocks = <Widget>[];
    void addBlock(Widget child) {
      if (blocks.isNotEmpty) {
        blocks.add(const SizedBox(height: 8));
      }
      blocks.add(child);
    }

    addBlock(
      Text(
        title,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16),
      ),
    );
    if (company != null) {
      addBlock(
        Text(
          company,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(color: Colors.black54, fontWeight: FontWeight.w600),
        ),
      );
    }
    if (hasMeta) {
      addBlock(_metaSection(location: location, category: category));
    }
    if (previewText != null || showChevron) {
      addBlock(_previewRow(previewText: previewText, isSalary: salary != null, showChevron: showChevron));
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
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  height: 44,
                  width: 44,
                  decoration: BoxDecoration(
                    color: AppTheme.primary.withOpacity(0.08),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.work_outline_rounded, color: AppTheme.primary),
                ),
                const SizedBox(width: 12),
                Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: blocks)),
                if (trailing != null) Padding(padding: const EdgeInsets.only(left: 8), child: trailing!),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _metaSection({String? location, String? category}) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final maxWidth = constraints.maxWidth;
        final widgets = <Widget>[];
        if (location != null) {
          widgets.add(_pill(Icons.place_outlined, location, maxWidth: maxWidth));
        }
        if (category != null) {
          if (widgets.isNotEmpty) {
            widgets.add(const SizedBox(height: 6));
          }
          widgets.add(_pill(Icons.category_outlined, category, maxWidth: maxWidth));
        }
        return Column(crossAxisAlignment: CrossAxisAlignment.start, children: widgets);
      },
    );
  }

  Widget _previewRow({required String? previewText, required bool isSalary, required bool showChevron}) {
    Widget preview;
    if (previewText == null) {
      preview = const SizedBox.shrink();
    } else if (isSalary) {
      preview = _salaryPill(previewText);
    } else {
      preview = Text(
        previewText,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(color: Colors.black87, fontWeight: FontWeight.w600),
      );
    }

    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(child: preview),
        if (showChevron) ...[
          const SizedBox(width: 8),
          const Icon(Icons.arrow_forward_ios_rounded, size: 16, color: Colors.black54),
        ],
      ],
    );
  }

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

  Widget _pill(IconData icon, String label, {required double maxWidth}) {
    return ConstrainedBox(
      constraints: BoxConstraints(maxWidth: maxWidth),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: const Color(0xFFF1F5F9),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: Colors.black54),
            const SizedBox(width: 4),
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

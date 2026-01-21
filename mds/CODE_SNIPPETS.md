# UI Redesign - Code Snippets & Examples

## Quick Reference Guide

Copy and paste these examples to quickly implement common patterns.

---

## 🎨 Theme Colors

### Using Theme Colors in Your Code

```dart
import 'package:job_recommender_app/theme/app_theme.dart';

// Access theme colors
Container(
  color: AppTheme.primary,  // Orange
  child: Text(
    'Hello',
    style: TextStyle(color: AppTheme.surface),  // White
  ),
)

// Use in decorations
decoration: BoxDecoration(
  color: AppTheme.muted,
  borderRadius: BorderRadius.circular(16),
  border: Border.all(color: AppTheme.divider),
  boxShadow: AppTheme.boxShadowSmall,
)

// Use gradients
decoration: BoxDecoration(
  gradient: AppTheme.gradientPrimary,
)
```

### Color Reference

```dart
// Primary Colors
AppTheme.primary        // #FF7F3F - Orange
AppTheme.primaryLight   // #FFBF99 - Light Orange
AppTheme.primaryDark    // #E65100 - Dark Orange

// Secondary
AppTheme.accent         // #5B5BFF - Purple
AppTheme.accentLight    // #7B7BFF - Light Purple

// Neutrals
AppTheme.background    // #FAFAFA - Off-white
AppTheme.surface       // #FFFFFF - White
AppTheme.muted         // #F5F5F5 - Light gray
AppTheme.mutedText     // #757575 - Gray text
AppTheme.divider       // #EEEEEE - Light border

// Status
AppTheme.success       // #4CAF50 - Green
AppTheme.warning       // #FFC107 - Orange
AppTheme.error         // #EF5350 - Red
```

---

## 📱 Reusable Components

### ModernJobCard

```dart
import 'package:job_recommender_app/widgets/modern_widgets.dart';

ModernJobCard(
  title: 'Senior Flutter Developer',
  company: 'Google',
  location: 'Mountain View, CA',
  category: 'Engineering',
  salary: '\$150K - \$200K',
  isSaved: false,
  applicantCount: 42,
  onTap: () {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => JobDetailsPage(job: job)),
    );
  },
  onSaveToggle: () async {
    await jobProvider.toggleSavedJob(job);
    setState(() {});
  },
)
```

### ModernSearchBar

```dart
final TextEditingController searchController = TextEditingController();

ModernSearchBar(
  controller: searchController,
  hintText: 'Search jobs, companies...',
  onChanged: (value) {
    // Called as user types (debounced in your code)
    performSearch(value);
  },
  onFilterTap: () {
    // Called when filter icon tapped
    setState(() => showFilters = !showFilters);
  },
  onClear: () {
    // Called when X icon tapped
    clearAllFilters();
  },
)
```

### FilterChipsGroup

```dart
final List<String> categories = [
  'All',
  'Engineering',
  'Design',
  'Product',
  'Marketing',
];

FilterChipsGroup(
  chips: categories,
  selectedChip: 'Engineering',
  onChipSelected: (selectedCategory) {
    setState(() {
      selectedChip = selectedCategory;
    });
    loadJobsByCategory(selectedCategory);
  },
)
```

### EmptyState

```dart
EmptyState(
  icon: Icons.inbox_rounded,
  title: 'No Saved Jobs Yet',
  subtitle: 'Jobs you save will appear here. Start exploring!',
  actionLabel: 'Browse Jobs',
  onAction: () {
    Navigator.pop(context);  // Return to home
  },
)
```

### LoadingJobCard

```dart
// For skeleton loading while fetching jobs
Column(
  children: List.generate(3, (_) => const LoadingJobCard()),
)
```

---

## 🎬 Animations

### Fade In Animation

```dart
import 'package:flutter/material.dart';

class MyFadingWidget extends StatefulWidget {
  @override
  State<MyFadingWidget> createState() => _MyFadingWidgetState();
}

class _MyFadingWidgetState extends State<MyFadingWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;
  late Animation<double> animation;

  @override
  void initState() {
    super.initState();
    controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    
    animation = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: controller, curve: Curves.easeInOut),
    );
    
    controller.forward();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: animation,
      child: Container(
        color: AppTheme.primary,
        child: const Text('Hello'),
      ),
    );
  }
}
```

### Scale On Tap

```dart
class ScalableCard extends StatefulWidget {
  const ScalableCard({required this.child});
  final Widget child;

  @override
  State<ScalableCard> createState() => _ScalableCardState();
}

class _ScalableCardState extends State<ScalableCard>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;

  @override
  void initState() {
    super.initState();
    controller = AnimationController(
      duration: const Duration(milliseconds: 200),
      vsync: this,
    );
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (_) => controller.forward(),
      onTapUp: (_) => controller.reverse(),
      onTapCancel: () => controller.reverse(),
      child: ScaleTransition(
        scale: Tween<double>(begin: 1.0, end: 0.98).animate(
          CurvedAnimation(parent: controller, curve: Curves.easeInOut),
        ),
        child: widget.child,
      ),
    );
  }
}
```

### Shimmer Loading

```dart
class ShimmerLoader extends StatefulWidget {
  @override
  State<ShimmerLoader> createState() => _ShimmerLoaderState();
}

class _ShimmerLoaderState extends State<ShimmerLoader>
    with SingleTickerProviderStateMixin {
  late AnimationController controller;

  @override
  void initState() {
    super.initState();
    controller = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, child) {
        return Container(
          decoration: BoxDecoration(
            gradient: LinearGradient(
              colors: [
                AppTheme.muted,
                AppTheme.muted.withOpacity(0.5),
                AppTheme.muted,
              ],
              stops: [0, controller.value, 1],
            ),
            borderRadius: BorderRadius.circular(8),
          ),
          height: 20,
          width: 150,
        );
      },
    );
  }
}
```

---

## 🧩 Custom Widgets

### Custom Button with Icon

```dart
ElevatedButton.icon(
  onPressed: () => applyForJob(),
  icon: const Icon(Icons.send_rounded),
  label: const Text('Apply Now'),
  style: ElevatedButton.styleFrom(
    backgroundColor: AppTheme.primary,
    minimumSize: const Size.fromHeight(52),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(12),
    ),
  ),
)
```

### Custom Info Card

```dart
Container(
  padding: const EdgeInsets.all(16),
  decoration: BoxDecoration(
    color: AppTheme.muted,
    borderRadius: BorderRadius.circular(12),
    border: Border.all(color: AppTheme.divider),
  ),
  child: Row(
    children: [
      Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: AppTheme.primary.withOpacity(0.1),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Icon(
          Icons.info_outline,
          color: AppTheme.primary,
          size: 20,
        ),
      ),
      const SizedBox(width: 12),
      Expanded(
        child: Text(
          'Important information about this job',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
      ),
    ],
  ),
)
```

### Custom Status Badge

```dart
Container(
  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
  decoration: BoxDecoration(
    color: isApplied ? AppTheme.success.withOpacity(0.1) : AppTheme.muted,
    borderRadius: BorderRadius.circular(6),
  ),
  child: Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Icon(
        isApplied ? Icons.check_circle : Icons.pending,
        size: 14,
        color: isApplied ? AppTheme.success : AppTheme.mutedText,
      ),
      const SizedBox(width: 4),
      Text(
        isApplied ? 'Applied' : 'Apply',
        style: Theme.of(context).textTheme.bodySmall?.copyWith(
          color: isApplied ? AppTheme.success : AppTheme.mutedText,
          fontWeight: FontWeight.w600,
        ),
      ),
    ],
  ),
)
```

---

## 📋 Page Layouts

### SliverAppBar with Custom Header

```dart
CustomScrollView(
  slivers: [
    SliverAppBar(
      expandedHeight: 120,
      pinned: true,
      backgroundColor: AppTheme.surface,
      flexibleSpace: FlexibleSpaceBar(
        background: Container(
          color: AppTheme.surface,
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.end,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Hello, User! 👋',
                    style: Theme.of(context).textTheme.bodyMedium),
                Text('Find your dream job',
                    style: Theme.of(context).textTheme.headlineSmall),
              ],
            ),
          ),
        ),
      ),
    ),
    // Content slivers here
  ],
)
```

### Sticky Bottom Action Bar

```dart
Scaffold(
  body: CustomScrollView(
    slivers: [
      // Your content
    ],
  ),
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
    child: Row(
      children: [
        Expanded(
          child: OutlinedButton(onPressed: () {}, child: const Text('Cancel')),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton(onPressed: () {}, child: const Text('Confirm')),
        ),
      ],
    ),
  ),
)
```

---

## 🔍 Common Patterns

### Search with Debounce

```dart
Timer? debounce;

@override
void initState() {
  super.initState();
  searchController.addListener(() {
    if (debounce?.isActive ?? false) debounce!.cancel();
    debounce = Timer(const Duration(milliseconds: 500), () {
      performSearch(searchController.text);
    });
  });
}

@override
void dispose() {
  debounce?.cancel();
  super.dispose();
}
```

### Pull to Refresh

```dart
RefreshIndicator(
  onRefresh: () async {
    await refreshData();
  },
  color: AppTheme.primary,
  child: ListView(
    children: [...],
  ),
)
```

### Error Handling with Snackbar

```dart
ScaffoldMessenger.of(context).showSnackBar(
  SnackBar(
    content: const Text('Operation failed'),
    duration: const Duration(seconds: 2),
    action: SnackBarAction(
      label: 'Retry',
      onPressed: () => retryOperation(),
    ),
    backgroundColor: AppTheme.error,
  ),
)
```

### Success Message

```dart
ScaffoldMessenger.of(context).showSnackBar(
  SnackBar(
    content: const Text('Job saved successfully! 🎉'),
    duration: const Duration(milliseconds: 800),
    backgroundColor: AppTheme.success,
  ),
)
```

---

## 🎯 Provider Integration

### Using with Provider

```dart
import 'package:provider/provider.dart';

// In your widget
final jobProvider = context.watch<JobProvider>();
final authProvider = context.watch<AuthProvider>();

// Update state
context.read<JobProvider>().loadJobs();

// Rebuild only when needed
Consumer<JobProvider>(
  builder: (context, jobs, child) {
    return Text('Jobs: ${jobs.jobs.length}');
  },
)
```

---

## 📐 Layout Spacing

### Standard Padding

```dart
// Page level
padding: const EdgeInsets.fromLTRB(16, 20, 16, 16)

// Card level
padding: const EdgeInsets.all(16)

// Between items
SizedBox(height: 12)

// Between sections
SizedBox(height: 28)
```

### Responsive Padding

```dart
Padding(
  padding: EdgeInsets.symmetric(
    horizontal: 16,
    vertical: MediaQuery.of(context).size.height * 0.02,
  ),
  child: widget,
)
```

---

## ✨ Styling Tricks

### Gradient Text

```dart
ShaderMask(
  shaderCallback: (bounds) => LinearGradient(
    colors: [AppTheme.primary, AppTheme.accent],
  ).createShader(bounds),
  child: Text(
    'Gradient Text',
    style: TextStyle(
      color: Colors.white,
      fontWeight: FontWeight.bold,
      fontSize: 24,
    ),
  ),
)
```

### Glassmorphism Effect

```dart
BackdropFilter(
  filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
  child: Container(
    decoration: BoxDecoration(
      color: Colors.white.withOpacity(0.1),
      borderRadius: BorderRadius.circular(16),
      border: Border.all(
        color: Colors.white.withOpacity(0.2),
      ),
    ),
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: widget,
    ),
  ),
)
```

### Elevated Card with Hover

```dart
MouseRegion(
  onEnter: (_) => setState(() => isHovered = true),
  onExit: (_) => setState(() => isHovered = false),
  child: AnimatedContainer(
    duration: const Duration(milliseconds: 200),
    transform: Matrix4.identity()..translate(0, isHovered ? -4 : 0),
    decoration: BoxDecoration(
      boxShadow: isHovered
          ? AppTheme.boxShadowLarge
          : AppTheme.boxShadowSmall,
    ),
    child: widget,
  ),
)
```

---

## 🧪 Testing

### Widget Test Example

```dart
testWidgets('ModernJobCard displays correctly', (WidgetTester tester) async {
  await tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: ModernJobCard(
          title: 'Test Job',
          company: 'Test Company',
          location: 'Remote',
          category: 'Engineering',
          salary: '\$100K',
          isSaved: false,
          onTap: () {},
          onSaveToggle: () {},
        ),
      ),
    ),
  );

  expect(find.text('Test Job'), findsOneWidget);
  expect(find.text('Test Company'), findsOneWidget);
  
  await tester.tap(find.byIcon(Icons.bookmark_border_rounded));
  await tester.pump();
});
```

---

## 📚 Additional Resources

- **Flutter Docs**: https://flutter.dev/docs
- **Material Design 3**: https://m3.material.io
- **Google Fonts**: https://fonts.google.com
- **Color Palette Generator**: https://coolors.co

---

**Version**: 1.0  
**Last Updated**: January 21, 2026

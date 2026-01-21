# Asset Recommendations for Firqah Lab UI Redesign

## Directory Structure

```
assets/
├── images/
│   ├── logo.png
│   ├── onboarding_hero.png
│   ├── illustrations/
│   │   ├── empty_state_jobs.png
│   │   ├── empty_state_saved.png
│   │   ├── empty_state_applications.png
│   │   └── success_animation.gif
│   └── company_logos/
│       ├── amazon.png
│       ├── google.png
│       ├── microsoft.png
│       └── ...
└── icons/
    └── (optional - for custom icons)
```

## Required Assets

### 1. App Logo (`assets/images/logo.png`)

**Specifications:**
- Format: PNG with transparency
- Minimum Size: 192x192px (will be scaled down for app use)
- Recommended: 512x512px for crisp scaling
- Safe Area: Leave 10% padding around logo
- Color: Use primary orange (`#FF7F3F`) or white variant

**How to Use:**
```dart
SizedBox(
  height: 60,
  child: Image.asset(
    'assets/images/logo.png',
    fit: BoxFit.contain,
  ),
)
```

**Where It Appears:**
- Onboarding screen header
- Login page
- App drawer/navigation

### 2. Onboarding Hero Image (`assets/images/onboarding_hero.png`)

**Specifications:**
- Format: PNG with transparency (recommended)
- Size: 1080x1080px minimum (square for flexibility)
- Aspect Ratio: 1:1 (square) or 16:9 (widescreen)
- Style: Modern, gradient, or abstract design
- Colors: Complement primary orange and accent purple

**Design Inspiration:**
- Clean, minimal geometric shapes
- Gradient overlays in orange/purple
- Floating elements or 3D perspective
- Abstract career/job-related imagery

**How to Use:**
```dart
Container(
  decoration: BoxDecoration(
    gradient: AppTheme.gradientBackground,
  ),
  child: Image.asset(
    'assets/images/onboarding_hero.png',
    fit: BoxFit.cover,
  ),
)
```

## Optional Illustration Assets

### 3. Empty State Illustrations

Create or source illustrations for:

#### `empty_state_jobs.png`
- **Use Case**: When no jobs are found
- **Size**: 300x300px
- **Style**: Friendly, minimal line art
- **Colors**: Grayscale with orange accents

#### `empty_state_saved.png`
- **Use Case**: When no saved jobs exist
- **Size**: 300x300px
- **Imagery**: Bookmark or heart icon with positive vibes

#### `empty_state_applications.png`
- **Use Case**: When no applications submitted
- **Size**: 300x300px
- **Imagery**: Paper, form, or checkmark elements

### 4. Success Animation (`success_animation.gif`)

**Specifications:**
- Format: GIF or Lottie JSON
- Duration: 1-2 seconds
- Size: 200x200px
- Colors: Primary orange and success green
- Loop: Once or loop 2-3 times

**When to Use:**
- After successful job application
- After saving a job
- After profile update

## Company Logo Assets (Optional)

Create a folder `assets/images/company_logos/` with company logos:

**Specifications:**
- Format: PNG with transparency
- Size: 60x60px (will be displayed in cards)
- Naming: `company_name.png` (lowercase, hyphenated)
- Style: Logo-only (no company name text)

**Recommended Companies to Include:**
- amazon.png
- google.png
- microsoft.png
- apple.png
- meta.png
- netflix.png
- tesla.png
- uber.png
- airbnb.png
- spotify.png

**How to Use in Code:**
```dart
SizedBox(
  width: 60,
  height: 60,
  child: Image.asset(
    'assets/images/company_logos/${companyName.toLowerCase()}.png',
    fit: BoxFit.contain,
    errorBuilder: (context, error, stackTrace) {
      // Fallback to letter avatar
      return Container(
        decoration: BoxDecoration(
          color: AppTheme.primary,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Center(
          child: Text(companyName[0].toUpperCase()),
        ),
      );
    },
  ),
)
```

## Asset File Sizes & Optimization

### Recommended File Sizes

| Asset | Format | Size (disk) | Dimensions |
|-------|--------|-----------|------------|
| Logo | PNG | 50-100 KB | 512x512px |
| Onboarding Hero | PNG | 100-200 KB | 1080x1080px |
| Empty State | PNG | 50-100 KB | 300x300px |
| Company Logo | PNG | 20-50 KB | 60x60px |
| Success Animation | GIF | 200-500 KB | 200x200px |

### Optimization Tips

1. **Use PNG for Static Images**
   - Use `pngquant` to reduce colors
   - Keep transparency only where needed

2. **Use WebP for Better Compression**
   - Supported on Android 4.0+, iOS 14+, Web
   - ~25% smaller than PNG
   - Fall back to PNG for older devices

3. **Use SVG for Icons (if possible)**
   - Vector format, infinitely scalable
   - Use `flutter_svg` package for SVG support
   - Smaller file sizes than raster graphics

4. **Lazy Load Company Logos**
   - Only load logos that are visible on screen
   - Cache downloaded logos locally

## Integration Instructions

### Step 1: Update pubspec.yaml

Uncomment and update the assets section:

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/images/logo.png
    - assets/images/onboarding_hero.png
    - assets/images/illustrations/
    - assets/images/company_logos/
```

### Step 2: Create Asset Directories

```bash
mkdir -p assets/images/illustrations
mkdir -p assets/images/company_logos
```

### Step 3: Add Assets

Place your PNG files in the respective directories.

### Step 4: Reference in Code

```dart
// Static asset
Image.asset('assets/images/logo.png')

// Asset with fallback
Image.asset(
  'assets/images/company_logos/amazon.png',
  errorBuilder: (context, error, stackTrace) {
    return SizedBox.square(
      dimension: 60,
      child: Container(
        color: AppTheme.muted,
        child: Center(child: Text('A')),
      ),
    );
  },
)
```

## Free Asset Resources

### Illustration Providers

1. **Undraw** (https://undraw.co)
   - Free, beautiful illustrations
   - Customizable colors (set to primary orange)
   - SVG format available

2. **Humaaans** (https://www.humaaans.com)
   - Character illustrations
   - Customizable poses and colors
   - Perfect for job/career themes

3. **Storyset** (https://storyset.com)
   - Animated illustrations
   - Multiple themes (career, business, success)
   - SVG and PNG formats

4. **Pexels** (https://www.pexels.com)
   - Free stock photos
   - Search for "job interview", "office", "career"

### Icon Providers

1. **Feather Icons** (https://feathericons.com)
   - Minimal, clean design
   - Matches Firqah Lab aesthetic
   - SVG format

2. **Heroicons** (https://heroicons.com)
   - Clean, modern icons
   - 24px and 20px sizes
   - MIT license

## Animation Assets

### Lottie Animations

For more complex animations, use the Lottie library:

1. **Add dependency** to `pubspec.yaml`:
```yaml
dependencies:
  lottie: ^2.4.0
```

2. **Download animations** from:
   - https://lottiefiles.com (search for "success", "loading", "checkmark")
   - Filter by free animations

3. **Add to project**:
```
assets/animations/
├── success.json
├── loading.json
└── ...
```

4. **Use in code**:
```dart
Lottie.asset('assets/animations/success.json')
```

## Design Tool Recommendations

### Create Your Own Assets

1. **Figma** (https://figma.com)
   - Free design tool
   - Create illustrations, logos, animations
   - Export as PNG, SVG, GIF

2. **Canva** (https://canva.com)
   - Easy drag-and-drop design
   - Templates for app design
   - Export as PNG/PDF

3. **Adobe XD** (https://www.adobe.com/products/xd.html)
   - Professional design tool
   - Free student/educator plans
   - Export in multiple formats

## Accessibility for Assets

### Image Descriptions

Always provide semantic descriptions:

```dart
Semantics(
  label: 'Success checkmark animation',
  child: Lottie.asset('assets/animations/success.json'),
)
```

### Content Descriptions

```dart
Image.asset(
  'assets/images/logo.png',
  semanticLabel: 'Job Recommender App Logo',
)
```

### Color Contrast

- Ensure logo is visible on both light and dark backgrounds
- Company logos should have sufficient contrast with card background
- Empty state illustrations should use colors from theme

## Platform-Specific Considerations

### Android

- Place assets in `assets/` directory
- Flutter automatically handles density scaling
- `Image.asset()` handles platform differences

### iOS

- Same approach as Android
- No additional configuration needed
- Vector-based assets (SVG) reduce app size

### Web

- Assets are served from web folder
- Use `Image.network()` for better performance in production
- Consider using CDN for large image files

## Testing Assets

### Quick Asset Test

Add this to verify assets load correctly:

```dart
// In your dev build
void testAssets() {
  debugPrint('Testing asset loading...');
  
  final imageProvider = AssetImage('assets/images/logo.png');
  imageProvider.resolve(ImageConfiguration.empty).addListener(
    ImageStreamListener((image, synchronousCall) {
      debugPrint('Logo asset loaded successfully');
    }),
  );
}
```

## Summary

| Asset | Priority | Size | Format | Created By |
|-------|----------|------|--------|-----------|
| App Logo | **High** | 512x512px | PNG | Designer/You |
| Onboarding Hero | Medium | 1080x1080px | PNG | Designer/Figma |
| Empty State Icons | Medium | 300x300px | PNG | Designer/Undraw |
| Company Logos | Low | 60x60px | PNG | Web sources |
| Animations | Low | - | JSON/GIF | Lottiefiles |

---

**Note**: Start with the logo and onboarding hero. These are the most visible and impactful assets. Company logos and animations can be added later as enhancement.

**Next Steps**:
1. Create or download the app logo
2. Design onboarding hero image (use Figma or Undraw)
3. Update `pubspec.yaml` with asset paths
4. Run `flutter pub get`
5. Test with `flutter run`

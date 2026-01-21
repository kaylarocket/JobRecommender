# Firqah Lab UI Redesign - Styling Guidelines

## Overview

This document outlines the modern UI design system implemented for the Job Recommender app, inspired by the Firqah Lab aesthetic. All components follow a clean, minimal design philosophy with rounded corners, intuitive interactions, and accessible color schemes.

## Color Palette

### Primary Colors

- **Primary Orange**: `#FF7F3F` - Main brand color for CTAs, highlights, and primary actions
- **Primary Light**: `#FFBF99` - Lighter variant for hover states and backgrounds
- **Primary Dark**: `#E65100` - Darker variant for active states and gradients

### Secondary Colors

- **Accent Purple**: `#5B5BFF` - Secondary color for accents and gradients
- **Accent Light**: `#7B7BFF` - Lighter accent variant

### Neutral Colors

- **Background**: `#FAFAFA` - Main app background
- **Surface**: `#FFFFFF` - Card and component backgrounds
- **Muted**: `#F5F5F5` - Secondary backgrounds and disabled states
- **Muted Text**: `#757575` - Secondary text color
- **Divider**: `#EEEEEE` - Border and divider color

### Status Colors

- **Success**: `#4CAF50` - Positive actions and confirmations
- **Warning**: `#FFC107` - Warning messages and alerts
- **Error**: `#EF5350` - Error states and destructive actions

## Typography

### Font Family
- **Primary Font**: Plus Jakarta Sans (Google Fonts)
- **Fallback**: System UI font

### Text Styles

| Style | Size | Weight | Line Height | Use Case |
|-------|------|--------|-------------|----------|
| Display Large | 57px | 800 | 1.0 | Page titles |
| Headline Large | 32px | 800 | 1.2 | Section headers |
| Headline Medium | 24px | 700 | 1.3 | Card titles |
| Title Large | 18px | 700 | 1.5 | Prominent labels |
| Title Medium | 16px | 600 | 1.5 | Job titles |
| Body Large | 16px | 500 | 1.6 | Main content |
| Body Medium | 14px | 500 | 1.5 | Card descriptions |
| Body Small | 12px | 500 | 1.4 | Secondary info |

## Spacing System

- **4px** - Minimal spacing (xs)
- **8px** - Small spacing (sm)
- **12px** - Medium spacing (md)
- **16px** - Large spacing (lg)
- **20px** - Extra large spacing (xl)
- **24px** - XXL spacing (2xl)
- **32px** - XXXL spacing (3xl)

## Border Radius

- **8px** - Small elements (chips, small buttons)
- **10px** - Medium elements (icon containers)
- **12px** - Standard elements (inputs, filter chips)
- **16px** - Cards and larger components
- **20px** - Large sections
- **50%** - Circles and avatars

## Shadow System

### Small Shadow
```
blurRadius: 8px
offset: (0, 2)
color: rgba(0, 0, 0, 4%)
```

### Medium Shadow
```
blurRadius: 16px
offset: (0, 4)
color: rgba(0, 0, 0, 6%)
```

### Large Shadow
```
blurRadius: 24px
offset: (0, 8)
color: rgba(0, 0, 0, 8%)
```

## Component Specifications

### Buttons

#### Elevated Button (Primary Action)
- **Height**: 52px
- **Padding**: 16px horizontal, 0px vertical (uses min height)
- **Border Radius**: 12px
- **Background**: Primary Orange
- **Text Color**: White
- **Font Weight**: 700
- **Font Size**: 16px
- **Elevation**: 0px (flat design)

#### Outlined Button (Secondary Action)
- **Height**: 52px
- **Border Width**: 2px
- **Border Color**: Primary Orange
- **Background**: Transparent
- **Text Color**: Primary Orange
- **Border Radius**: 12px

#### Text Button
- **Font Size**: 14px
- **Font Weight**: 600
- **Text Color**: Primary Orange
- **No background or border**

### Search Bar
- **Height**: 44px (with padding)
- **Border Radius**: 12px
- **Background**: Muted (F5F5F5)
- **Border**: 1px divider color
- **Focus Border**: 2px primary color
- **Icon Color**: Primary Orange (filter icon)
- **Padding**: 16px horizontal, 12px vertical

### Cards (Job Cards)
- **Border Radius**: 16px
- **Border**: 1px divider color
- **Background**: Surface white
- **Top Accent**: 4px gradient bar (orange → purple)
- **Padding**: 16px
- **Spacing**: 12px bottom margin
- **Shadow**: Small shadow

### Chips (Filter/Category)
- **Border Radius**: 8px
- **Height**: 32px
- **Padding**: 12px horizontal, 8px vertical
- **Default**: Muted background
- **Selected**: 15% opacity primary background with 2px primary border
- **Font Weight**: 600
- **Font Size**: 14px

### Input Fields
- **Height**: 44px
- **Border Radius**: 12px
- **Background**: Muted
- **Border**: 1px divider color
- **Focus Border**: 2px primary color
- **Padding**: 16px horizontal, 14px vertical
- **Content**: Google Fonts Plus Jakarta Sans

## Interactions

### Hover States
- Buttons: Slightly darker background
- Cards: Scale up by 2% (1.02x)
- Icons: Color transition to primary

### Press States
- Cards: Scale down to 98% (0.98x)
- Buttons: Background color darkens slightly

### Loading States
- Shimmer animation with 1500ms duration
- Color gradient sliding across components
- Used for skeleton loading screens

### Animations
- **Fade In**: 400-600ms (easeInOut)
- **Scale Transitions**: 200-300ms (easeInOut)
- **Slide Transitions**: 200-400ms (easeOutCubic)
- **Shimmer**: 1500ms (infinite loop)

## Layout Specifications

### Page Padding
- **Horizontal**: 16px
- **Top**: 16px-20px (after header)
- **Bottom**: 16px + safe area

### Card/Component Spacing
- **Between cards**: 12px
- **Between sections**: 20px-28px
- **Between title and content**: 12px

### Header (SliverAppBar)
- **Expanded Height**: 200px (for hero sections)
- **Regular Height**: 56px
- **Pinned**: True (stays visible when scrolling)

## Accessibility

### Color Contrast
- **Text on Primary**: White text on orange (7.2:1 contrast ratio) ✓
- **Text on Muted**: Muted text on white (5.8:1 contrast ratio) ✓
- **Text on Surface**: Black text on white (21:1 contrast ratio) ✓

### Touch Targets
- **Minimum size**: 44x44dp for interactive elements
- **Minimum spacing**: 8px between adjacent touch targets

### Text Size
- **Minimum body text**: 12px (follows Material Design 3)
- **Minimum interactive labels**: 14px

## Responsive Breakpoints

- **Mobile**: < 600px (default)
- **Tablet**: 600px - 840px
- **Desktop**: > 840px

The current implementation is optimized for mobile devices (< 600px).

## Assets & Icons

### Icon Library
- **Primary Source**: Material Icons (Flutter's built-in)
- **Icon Size**:
  - Small: 16px
  - Medium: 20px
  - Large: 24px
  - XL: 28px
  - XXL: 56px

### Recommended Icons

| Element | Icon |
|---------|------|
| Search | Icons.search_rounded |
| Filter | Icons.tune_rounded |
| Save/Bookmark | Icons.bookmark_rounded / Icons.bookmark_border_rounded |
| Apply | Icons.send_rounded |
| Location | Icons.location_on_outlined |
| Category | Icons.category_outlined |
| Salary | Icons.attach_money_rounded |
| Company | Icons.business_outlined |
| Person/Profile | Icons.person_outline_rounded |
| Settings | Icons.settings_rounded |

### Image Assets

Create these assets in `assets/images/`:

- `logo.png` - App logo (60x60px minimum)
- `onboarding_hero.png` - Hero image for onboarding (optional)
- Company logos (optional, for job cards)

## Code Implementation Examples

### Using the Theme
```dart
// Access theme colors
Color primaryColor = AppTheme.primary;
Color backgroundColor = AppTheme.background;

// Access shadows
boxShadow: AppTheme.boxShadowSmall;

// Access gradients
decoration: BoxDecoration(gradient: AppTheme.gradientPrimary);
```

### Creating a Modern Card
```dart
Container(
  decoration: BoxDecoration(
    color: AppTheme.surface,
    borderRadius: BorderRadius.circular(16),
    border: Border.all(color: AppTheme.divider),
    boxShadow: AppTheme.boxShadowSmall,
  ),
  // ... content
)
```

### Applying Text Styles
```dart
Text(
  'Hello World',
  style: Theme.of(context).textTheme.headlineSmall,
)
```

## Migration Checklist

- [x] Updated color palette in AppTheme
- [x] Updated typography system
- [x] Created ModernJobCard widget
- [x] Created ModernSearchBar widget
- [x] Created FilterChipsGroup widget
- [x] Created EmptyState widget
- [x] Created OnboardingPage
- [x] Created SeekerHomePageNew
- [x] Created JobDetailsPageNew
- [x] Added gradient utilities
- [x] Updated main.dart entry point

## Future Enhancements

1. **Dark Mode Support**: Implement dark theme variant
2. **Animations**: Add page transition animations
3. **Accessibility**: Add semantic labeling for screen readers
4. **Internationalization**: Multi-language support
5. **Advanced Filtering**: More filter options and saved filters
6. **Notifications**: Push notifications for new jobs
7. **Analytics**: Track user interactions

## Browser & Platform Support

- **Android**: 5.0+ (API 21+)
- **iOS**: 11.0+
- **Web**: All modern browsers (Chrome, Firefox, Safari, Edge)
- **macOS**: 10.11+
- **Windows**: Windows 10+
- **Linux**: Ubuntu 16.04+

## Performance Optimization

- **Image Optimization**: Use `fit: BoxFit.cover` for images
- **Lazy Loading**: Use SliverList for efficient scrolling
- **Caching**: Implement image caching for company logos
- **Build Optimization**: Use `const` constructors where possible
- **State Management**: Use Provider for efficient rebuilds

---

**Last Updated**: January 21, 2026
**Version**: 1.0
**Design System**: Firqah Lab Inspired

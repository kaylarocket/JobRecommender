# Recruiter Pages Documentation

## Overview

Four new recruiter pages have been created for the job application app, providing a complete recruiter management system. All pages follow a consistent, modern UI design with a mobile-first approach.

## Design System

**Colors:**
- Primary: `#4F46E5` (Indigo/Purple)
- Muted Background: `#F1F5F9`
- Border: `#E2E8F0`
- Success: `#15803D` (Green)
- Danger: `#DC2626` (Red)
- Info: `#2563EB` (Blue)

**Typography:**
- Font: Plus Jakarta Sans
- Heading: 16-20px, Weight 700 (Bold)
- Body: 14px, Weight 400-500
- Small: 12-13px, Weight 500-600

**Components:**
- Rounded cards (14-16px border radius)
- Soft shadows (0.02-0.03 opacity)
- Consistent padding (14-20px)
- Status badges with color-coded backgrounds

---

## Pages

### 1. **Manage Jobs** (`manage_jobs_page.dart`)

**Purpose:** Display all posted jobs with status management and editing capabilities.

**Features:**
- Filter tabs: All, Active, Closed
- Job cards showing:
  - Job title
  - Location
  - Status badge (Active/Closed)
  - Applicant count
  - Posted date
- Action buttons: Edit, Close
- Empty state when no jobs exist

**Key Components:**
- `FilterChip` for status filtering
- `_jobManagementCard()` - displays individual job with actions
- `_statBadge()` - shows statistics with icons

**Navigation:**
```dart
Navigator.push(context, MaterialPageRoute(builder: (_) => const ManageJobsPage()))
```

---

### 2. **Applicants List** (`applicants_list_page.dart`)

**Purpose:** Display all applicants grouped by job with status filtering.

**Features:**
- Status tabs: All, New, Shortlisted, Rejected
- Applicants grouped by job title
- Applicant cards showing:
  - Applicant avatar with initials
  - Name and application date
  - Skills (truncated)
  - Status badge with color coding
- Quick "View" button to open detailed profile

**Status Colors:**
- New: Blue (`#DBEAFE` bg, `#2563EB` text)
- Shortlisted: Green (`#DCFCE7` bg, `#15803D` text)
- Rejected: Red (`#FEE2E2` bg, `#DC2626` text)

**Key Components:**
- `_applicantCard()` - displays individual applicant
- `_getStatusColor()` - returns color based on status
- `_getStatusBgColor()` - returns background color based on status

**Navigation:**
```dart
Navigator.push(
  context,
  MaterialPageRoute(builder: (_) => ApplicantDetailPage(applicant: applicant)),
)
```

---

### 3. **Applicant Detail** (`applicant_detail_page.dart`)

**Purpose:** Full-screen comprehensive profile of an applicant.

**Features:**
- Header section with:
  - Avatar with initials
  - Full name
  - Application date
  - Current status badge
- Skills section (displayed as tags)
- About section (bio/summary)
- Resume file card with:
  - Preview button
  - Download button
  - File size indicator
- Education timeline:
  - Degree
  - Institution
  - Graduation year
- Experience timeline:
  - Job title
  - Company
  - Duration
  - Description
- Portfolio links:
  - GitHub profile
  - Personal website
  - Other portfolio items
- Contact information:
  - Email
  - Phone number
- Action buttons:
  - Reject (with confirmation dialog)
  - Schedule Interview

**Key Components:**
- `_buildFileCard()` - resume file display
- `_buildEducationCard()` - education entry
- `_buildExperienceCard()` - work experience entry
- `_buildLinkCard()` - portfolio/external links
- `_buildContactCard()` - contact information

**Navigation:**
```dart
Navigator.push(
  context,
  MaterialPageRoute(builder: (_) => const ApplicantDetailPage(applicant: applicant)),
)
```

---

### 4. **Company Profile** (`company_profile_page.dart`)

**Purpose:** Display and edit company information.

**Features:**

**Display Mode:**
- Company logo/icon (100x100)
- Company name
- Industry badge
- Description
- Details section:
  - Location
  - Website
  - Founded year
  - Team size
- Contact information section:
  - Email (clickable)
  - Phone (clickable)
- Edit button

**Edit Mode:**
- Logo upload/change
- Editable fields:
  - Company name
  - Description (3-line textarea)
  - Location
  - Website
  - Industry
  - Email
  - Phone
- Save/Cancel buttons

**Key Components:**
- `_buildDisplayMode()` - read-only company view
- `_buildEditMode()` - editable company form
- `_buildDetailItem()` - detail card (display mode)
- `_buildContactItem()` - contact card (clickable)
- `_buildTextField()` - text input fields (edit mode)

**Navigation:**
```dart
Navigator.push(context, MaterialPageRoute(builder: (_) => const CompanyProfilePage()))
```

---

## Updated Components

### Recruiter Dashboard (`recruiter_dashboard_page.dart`)

**Added:** Quick access navigation cards to all new pages:

```dart
Row(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
  children: [
    _quickActionCard(...), // Manage Jobs
    _quickActionCard(...), // All Applicants
    _quickActionCard(...), // Company Info
  ],
)
```

**New Helper Method:**
- `_quickActionCard()` - creates navigable action cards with icon and label

---

## Common UI Patterns

### Empty State
```dart
Widget _emptyState() {
  return ListView(
    padding: const EdgeInsets.all(16),
    children: [
      Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFE2E8F0)),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppTheme.muted,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(...),
            ),
            const SizedBox(height: 16),
            Text('Empty state title'),
            Text('Empty state subtitle'),
          ],
        ),
      ),
    ],
  );
}
```

### Card with Shadow
```dart
Container(
  padding: const EdgeInsets.all(16),
  decoration: BoxDecoration(
    color: Colors.white,
    borderRadius: BorderRadius.circular(16),
    border: Border.all(color: const Color(0xFFE2E8F0)),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withOpacity(0.03),
        blurRadius: 8,
        offset: const Offset(0, 2),
      ),
    ],
  ),
  child: // content
)
```

### Status Badge
```dart
Container(
  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
  decoration: BoxDecoration(
    color: statusBgColor, // e.g., Color(0xFFDCFCE7)
    borderRadius: BorderRadius.circular(8),
  ),
  child: Text(
    status,
    style: TextStyle(
      fontSize: 12,
      fontWeight: FontWeight.w600,
      color: statusColor, // e.g., Color(0xFF15803D)
    ),
  ),
)
```

---

## State Management

All pages use:
- `Provider` for job data (existing integration)
- `StatefulWidget` for local state (editing, filtering)
- `ScaffoldMessenger` for toast notifications

Example state management in Manage Jobs:
```dart
class _ManageJobsPageState extends State<ManageJobsPage> {
  String _filterStatus = 'All';

  final filteredJobs = _filterStatus == 'All'
      ? jobs.postedJobs
      : jobs.postedJobs.where((job) => /* filter logic */).toList();
}
```

---

## Integration Points

### With Existing Code

1. **Auth Provider** - Used in dashboard for user name display
2. **Job Provider** - Used for job listings and applicant data
3. **Theme** - Uses `AppTheme.primary` and `AppTheme.muted` colors
4. **Navigation** - Integrated with existing navigation system

### Mock Data

Applicants List Page includes mock data:
```dart
final List<Map<String, dynamic>> _mockApplicants = [
  {
    'id': '1',
    'name': 'Alex Tan',
    'jobTitle': 'Senior Flutter Developer',
    'status': 'New',
    'appliedDate': '2 hours ago',
    'skills': 'Flutter, Dart, Firebase',
  },
  // ... more applicants
];
```

---

## Future Enhancements

1. **Backend Integration:**
   - Connect to actual job listings API
   - Fetch real applicant data from database
   - Implement actual file upload for resume/logo

2. **Advanced Features:**
   - Search and advanced filtering
   - Bulk actions (reject multiple, send messages)
   - Interview scheduling integration
   - Email notifications

3. **Analytics:**
   - Job performance metrics
   - Applicant conversion funnel
   - Time-to-hire dashboard

4. **Messaging:**
   - In-app messaging with applicants
   - Interview scheduling
   - Feedback/comments on applicants

---

## File Structure

```
lib/pages/recruiter/
├── recruiter_dashboard_page.dart      (Updated: Added navigation)
├── manage_jobs_page.dart              (New: Job management)
├── applicants_list_page.dart          (New: Applicant listings)
├── applicant_detail_page.dart         (New: Detailed profiles)
├── company_profile_page.dart          (New: Company info)
├── applicants_page.dart               (Existing: Legacy version)
└── post_job_page.dart                 (Existing: Job posting)
```

---

## Testing Checklist

- [ ] Manage Jobs - filter by status (All, Active, Closed)
- [ ] Manage Jobs - edit/close job actions
- [ ] Applicants List - filter by status
- [ ] Applicants List - view applicant details
- [ ] Applicant Detail - all sections display (about, resume, education, etc.)
- [ ] Applicant Detail - reject and schedule interview actions
- [ ] Company Profile - display mode shows all information
- [ ] Company Profile - edit mode allows field modification
- [ ] Recruiter Dashboard - quick access cards navigate correctly
- [ ] All pages - responsive design on different screen sizes
- [ ] All pages - smooth animations and transitions

---

## Notes

- All pages follow Material 3 design guidelines
- Mobile-first responsive design
- Consistent use of `AppTheme` colors and styles
- Proper error handling with `ScaffoldMessenger`
- Empty states provided for all list views

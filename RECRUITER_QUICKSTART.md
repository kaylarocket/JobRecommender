# Quick Start: Recruiter Pages Integration

## Overview
This guide shows how to integrate and navigate between the new recruiter pages.

## Quick Navigation Examples

### From Recruiter Dashboard
The dashboard now includes three quick-access cards:

```dart
// These are automatically available in the RecruiterDashboardPage
Row(
  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
  children: [
    _quickActionCard(
      context,
      icon: Icons.work_outline,
      label: 'Manage Jobs',
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const ManageJobsPage()),
      ),
    ),
    _quickActionCard(
      context,
      icon: Icons.people_outline,
      label: 'All Applicants',
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const ApplicantsListPage()),
      ),
    ),
    _quickActionCard(
      context,
      icon: Icons.business_outlined,
      label: 'Company Info',
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const CompanyProfilePage()),
      ),
    ),
  ],
)
```

## Page Navigation Flow

```
RecruiterDashboardPage (Home)
├── ManageJobsPage
│   └── Edit/Close Job Actions
├── ApplicantsListPage
│   └── ApplicantDetailPage (Full-screen view)
├── CompanyProfilePage
│   └── Edit Company Info
└── ApplicantsPage (Legacy - per job)
```

## Adding to App Drawer/Navigation

To add these pages to a side drawer or bottom navigation:

```dart
// In your main navigation menu
ListTile(
  leading: const Icon(Icons.work_outline),
  title: const Text('Manage Jobs'),
  onTap: () {
    Navigator.pop(context); // Close drawer
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ManageJobsPage()),
    );
  },
),
ListTile(
  leading: const Icon(Icons.people_outline),
  title: const Text('Applicants'),
  onTap: () {
    Navigator.pop(context);
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ApplicantsListPage()),
    );
  },
),
ListTile(
  leading: const Icon(Icons.business_outlined),
  title: const Text('Company Profile'),
  onTap: () {
    Navigator.pop(context);
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const CompanyProfilePage()),
    );
  },
),
```

## Importing the Pages

```dart
import 'pages/recruiter/manage_jobs_page.dart';
import 'pages/recruiter/applicants_list_page.dart';
import 'pages/recruiter/applicant_detail_page.dart';
import 'pages/recruiter/company_profile_page.dart';
```

## Data Flow

### Manage Jobs Page
```
User filters jobs → StatefulWidget updates _filterStatus
                  → filteredJobs list rebuilt
                  → JobCards displayed with edit/close actions
                  → Actions show SnackBar feedback
```

### Applicants List Page
```
Initial load → _mockApplicants loaded
            → Grouped by job title
            
User filters → _selectedStatus updated
            → Filtered list rebuilt
            
User clicks "View" → Navigate to ApplicantDetailPage
                  → Pass applicant data
```

### Applicant Detail Page
```
Receive applicant Map
↓
Display all sections:
├── Profile header (avatar, name, date)
├── Skills (as tags)
├── About section
├── Resume file (with download/preview)
├── Education timeline
├── Experience timeline
├── Portfolio links
├── Contact info
└── Action buttons
    ├── Reject (with confirmation)
    └── Schedule Interview
```

### Company Profile Page
```
Load company data from state
↓
Display mode: Read-only company info
Edit mode: Editable form fields
          ↓
          Save changes → Update state
          Cancel → Revert to display mode
```

## Mock Data Structure

Applicants List uses this data structure:

```dart
final List<Map<String, dynamic>> _mockApplicants = [
  {
    'id': '1',                           // Unique identifier
    'name': 'Alex Tan',                  // Applicant name
    'jobTitle': 'Senior Flutter Developer', // Position applied for
    'status': 'New',                     // New, Shortlisted, Rejected
    'appliedDate': '2 hours ago',        // Relative date
    'skills': 'Flutter, Dart, Firebase', // Comma-separated skills
  },
  // ... more applicants
];
```

Company Profile uses this structure:

```dart
final Map<String, String> _companyData = {
  'name': 'TechVision Solutions',
  'logo': '🏢',
  'description': 'Company description...',
  'location': 'Singapore, Singapore',
  'website': 'www.techvision.com',
  'industry': 'Technology / Software Development',
  'founded': '2018',
  'employees': '120-150',
  'email': 'careers@techvision.com',
  'phone': '+65 6234 5678',
};
```

## Status Constants

### Applicant Status Colors
```dart
// Status Badge Colors
New:         Background: #DBEAFE, Text: #2563EB
Shortlisted: Background: #DCFCE7, Text: #15803D  
Rejected:    Background: #FEE2E2, Text: #DC2626

// Job Status
Active:      Background: #DCFCE7, Text: #15803D
Closed:      Background: #FEE2E2, Text: #DC2626
```

## Common Tasks

### Refresh Job List
```dart
// In ManageJobsPage, if jobs change:
setState(() {
  _filterStatus = _filterStatus; // Trigger rebuild
});
```

### Filter Applicants
```dart
// Change status filter
setState(() => _selectedStatus = 'Shortlisted');

// Automatically rebuilds filtered list
final filteredApplicants = _selectedStatus == 'All'
    ? _mockApplicants
    : _mockApplicants.where((a) => a['status'] == _selectedStatus).toList();
```

### Save Company Changes
```dart
// In CompanyProfilePage edit mode:
void _saveChanges() {
  setState(() {
    _companyData['name'] = _nameController.text;
    _companyData['description'] = _descriptionController.text;
    // ... other fields
    _isEditing = false;
  });
  ScaffoldMessenger.of(context).showSnackBar(
    const SnackBar(content: Text('Company profile updated')),
  );
}
```

### Navigate with Data
```dart
// From Applicants List to Detail
Navigator.push(
  context,
  MaterialPageRoute(
    builder: (_) => ApplicantDetailPage(applicant: applicant),
  ),
);

// In ApplicantDetailPage, access data:
@override
Widget build(BuildContext context) {
  return Text(applicant['name']); // Use applicant map
}
```

## Backend Integration Steps

1. **Replace mock data in ApplicantsListPage:**
   ```dart
   // OLD: Load from _mockApplicants
   // NEW: Load from API
   Future<void> _loadApplicants() async {
     final applicants = await ApiService.getApplicants();
     setState(() => _mockApplicants = applicants);
   }
   
   @override
   void initState() {
     super.initState();
     _loadApplicants();
   }
   ```

2. **Replace mock data in CompanyProfilePage:**
   ```dart
   // OLD: Initialize with hardcoded data
   // NEW: Load from API
   @override
   void initState() {
     super.initState();
     _loadCompanyData();
   }
   
   Future<void> _loadCompanyData() async {
     final data = await ApiService.getCompanyProfile();
     setState(() => _companyData = data);
   }
   ```

3. **Connect ManageJobsPage to JobProvider:**
   ```dart
   // Already integrated! Uses:
   final jobs = context.watch<JobProvider>();
   
   // Just filter and display:
   final filteredJobs = jobs.postedJobs; // Already available
   ```

4. **Implement actual actions:**
   ```dart
   // Replace SnackBar with actual API calls
   
   // OLD: ScaffoldMessenger.of(context).showSnackBar(...)
   // NEW:
   onPressed: () async {
     try {
       await ApiService.closeJob(job.id);
       ScaffoldMessenger.of(context).showSnackBar(
         const SnackBar(content: Text('Job closed')),
       );
       // Refresh jobs list
       context.read<JobProvider>().refreshJobs();
     } catch (e) {
       ScaffoldMessenger.of(context).showSnackBar(
         SnackBar(content: Text('Error: $e')),
       );
     }
   },
   ```

## Testing Tips

1. **Test filtering:**
   - Click status tabs
   - Verify list updates
   - Check empty states

2. **Test navigation:**
   - Click view buttons
   - Verify data passes correctly
   - Test back navigation

3. **Test editing:**
   - Switch to edit mode
   - Modify fields
   - Test save and cancel

4. **Test responsiveness:**
   - Test on different screen sizes
   - Check card layouts on small screens
   - Verify text doesn't overflow

## Troubleshooting

**Issue:** Pages not appearing in navigation
- ✅ Check imports are correct
- ✅ Verify Navigator.push() syntax
- ✅ Check MaterialPageRoute wrapping

**Issue:** Data not displaying
- ✅ Check mock data is initialized
- ✅ Verify variable names in build()
- ✅ Check TextEditingController initialization

**Issue:** Status badges wrong colors
- ✅ Verify status value matches ('New', 'Shortlisted', 'Rejected')
- ✅ Check _getStatusColor() function
- ✅ Test with actual status values

**Issue:** Edit mode not working
- ✅ Check TextEditingController initialization in initState()
- ✅ Verify dispose() is called
- ✅ Test save button functionality

## Next Steps

1. ✅ UI fully implemented
2. ⏳ Connect to backend APIs
3. ⏳ Replace mock data
4. ⏳ Add real authentication
5. ⏳ Implement file uploads
6. ⏳ Add analytics tracking
7. ⏳ Deploy to production

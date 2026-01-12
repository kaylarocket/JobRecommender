# Recruiter Pages - Visual Guide

## Page Layouts and Flows

### 1. MANAGE JOBS PAGE

```
┌─────────────────────────────────┐
│  Manage Jobs          [🔙]     │
└─────────────────────────────────┘

┌─ Filter Tabs ──────────────────┐
│ [All] [Active] [Closed]        │
└────────────────────────────────┘

┌─ Job Card ─────────────────────┐
│ Senior Flutter Developer    │✓│ Active
│ Singapore                      │
│ 👥 12 applicants   📅 5 days ago
│                                │
│ ┌─ Edit ──┬─ Close ──┐       │
│ └──────────┴──────────┘       │
└────────────────────────────────┘

┌─ Job Card ─────────────────────┐
│ FastAPI Backend Engineer    │✓│ Active
│ Remote                         │
│ 👥 8 applicants    📅 1 day ago
│                                │
│ ┌─ Edit ──┬─ Close ──┐       │
│ └──────────┴──────────┘       │
└────────────────────────────────┘
```

**Key Features:**
- Tab-based filtering (All/Active/Closed)
- Status badges (color-coded)
- Applicant count & posted date
- Edit & Close action buttons
- Job cards with soft shadows

---

### 2. APPLICANTS LIST PAGE

```
┌─────────────────────────────────┐
│  Applicants               [🔙]  │
└─────────────────────────────────┘

┌─ Status Tabs ──────────────────┐
│ [All] [New] [Shortlisted] [Rejected]
└────────────────────────────────┘

📌 Senior Flutter Developer
┌─ Applicant Card ──────────────┐
│ [A] Alex Tan         2 hrs ago │
│     Flutter, Dart, Firebase   │
│                    [View] →    │
│                        🆕 New  │
└────────────────────────────────┘

┌─ Applicant Card ──────────────┐
│ [M] Maya Lee         1 day ago │
│     Flutter, Dart, REST APIs  │
│                    [View] →    │
│               ✅ Shortlisted   │
└────────────────────────────────┘

📌 FastAPI Backend Engineer
┌─ Applicant Card ──────────────┐
│ [J] John Doe         3 hrs ago │
│     Python, FastAPI, PostgreSQL│
│                    [View] →    │
│                        🆕 New  │
└────────────────────────────────┘
```

**Key Features:**
- Status filter tabs (All/New/Shortlisted/Rejected)
- Grouped by job title
- Avatar with initials
- Skills preview
- Color-coded status badges
- "View" button for details

---

### 3. APPLICANT DETAIL PAGE

```
┌──────────────────────────────────┐
│  Applicant Profile        [🔙]  │
└──────────────────────────────────┘

┌──────── Profile Header ────────┐
│         [A] Alex Tan          │
│    Applied 2 hours ago        │
│      ✅ Shortlisted           │
└───────────────────────────────┘

📌 Skills
┌─────────────────────────────────┐
│ [Flutter] [Dart] [Firebase]    │
│ [REST APIs] [Testing]          │
└─────────────────────────────────┘

📌 About
┌─ Description ─────────────────┐
│ Experienced Flutter developer  │
│ with 5+ years in mobile       │
│ development. Specialized in   │
│ cross-platform app...         │
└───────────────────────────────┘

📌 Resume
┌─ File Card ───────────────────┐
│ [📄] Alex_Tan_Resume.pdf  245KB│
│      [👁️ Preview] [⬇️ Download] │
└───────────────────────────────┘

📌 Education
┌─ Education Card ──────────────┐
│ [🎓] BS Computer Science      │
│     National University  2018 │
└───────────────────────────────┘
┌─ Education Card ──────────────┐
│ [🎓] Flutter Certification    │
│     Google Academy       2021 │
└───────────────────────────────┘

📌 Experience
┌─ Experience Card ─────────────┐
│ [💼] Senior Flutter Developer │
│     Tech Innovations Inc.    │
│     2021 - Present           │
│     Led mobile team, mentored │
│     3 junior developers      │
└───────────────────────────────┘

📌 Portfolio
┌─ Link Card ───────────────────┐
│ [🔗] GitHub Profile           │
│     github.com/alextan   → │
└───────────────────────────────┘

┌────────────────────────────────┐
│ [Reject] │ [Schedule Interview]│
└────────────────────────────────┘
```

**Key Features:**
- Full-screen detailed view
- Multiple sections (About, Resume, Education, Experience, Portfolio)
- File preview/download for resume
- Clickable contact info
- Action buttons (Reject, Schedule Interview)
- Timeline-style experience display

---

### 4. COMPANY PROFILE PAGE

#### Display Mode
```
┌─────────────────────────────────┐
│  Company Profile  [Edit] [🔙]  │
└─────────────────────────────────┘

┌──── Company Header ────────────┐
│            🏢                  │
│     TechVision Solutions       │
│    Technology / Software       │
│  Innovative tech company focused│
│  on building cutting-edge      │
│  mobile and web solutions...   │
└────────────────────────────────┘

📌 Company Details
┌─ Detail Card ──────────────────┐
│ [📍] Location                  │
│      Singapore, Singapore     │
└────────────────────────────────┘
┌─ Detail Card ──────────────────┐
│ [🌐] Website                   │
│      www.techvision.com       │
└────────────────────────────────┘
┌─ Detail Card ──────────────────┐
│ [📅] Founded                   │
│      2018                     │
└────────────────────────────────┘
┌─ Detail Card ──────────────────┐
│ [👥] Team Size                 │
│      120-150 employees        │
└────────────────────────────────┘

📌 Contact Information
┌─ Contact Card ─────────────────┐
│ [📧] Email                     │
│      careers@techvision.com   │
└────────────────────────────────┘
┌─ Contact Card ─────────────────┐
│ [📱] Phone                     │
│      +65 6234 5678      → │
└────────────────────────────────┘
```

#### Edit Mode
```
┌─────────────────────────────────┐
│  Company Profile  [Save] [🔙]  │
└─────────────────────────────────┘

┌──── Logo Upload ───────────────┐
│            🏢                  │
│        (Tap to change)         │
└────────────────────────────────┘

📌 Company Information
┌─ Text Field ───────────────────┐
│ 🏢 Company Name               │
│ ┌─────────────────────────┐   │
│ │ TechVision Solutions    │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

┌─ Text Field ───────────────────┐
│ 📝 Description (Multi-line)   │
│ ┌─────────────────────────┐   │
│ │ Innovative tech company│   │
│ │ focused on building... │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

┌─ Text Field ───────────────────┐
│ 📍 Location                   │
│ ┌─────────────────────────┐   │
│ │ Singapore, Singapore    │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

┌─ Text Field ───────────────────┐
│ 🌐 Website                    │
│ ┌─────────────────────────┐   │
│ │ www.techvision.com      │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

📌 Contact Information
┌─ Text Field ───────────────────┐
│ 📧 Email                      │
│ ┌─────────────────────────┐   │
│ │ careers@techvision.com  │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

┌─ Text Field ───────────────────┐
│ 📱 Phone                      │
│ ┌─────────────────────────┐   │
│ │ +65 6234 5678          │   │
│ └─────────────────────────┘   │
└────────────────────────────────┘

┌──────────────────────────────────┐
│ [Cancel] │ [Save Changes]       │
└──────────────────────────────────┘
```

**Key Features:**
- Toggle between display and edit modes
- Company header with logo/icon
- Organized sections (Details, Contact)
- Editable text fields with proper spacing
- Save/Cancel buttons
- Logo upload functionality

---

### 5. RECRUITER DASHBOARD (Updated)

```
┌─────────────────────────────────┐
│  Recruiter workspace  [➕] [🔙] │
└─────────────────────────────────┘

Welcome back, John
Manage your postings and applicants

[Post a Job]

┌─ Stats Row ─────────────────────┐
│ [💼] Open roles  │ [📥] Applications
│      5           │      12
└─────────────────────────────────┘

┌─ Quick Actions ─────────────────┐
│ [💼]      [👥]      [🏢]       │
│ Manage   All      Company     │
│  Jobs  Applicants  Info      │
└─────────────────────────────────┘

📌 Posted Jobs
┌─ Job Card ─────────────────────┐
│ Senior Flutter Developer   ✅   │
│ Singapore                      │
│ 👥 12 applicants   📅 5 days   │
│ [View Applicants] ✓            │
└────────────────────────────────┘

📌 Recent Applicants
┌─ Applicant Card ──────────────┐
│ [A] Alex Tan        2 hrs ago  │
│     New applicant              │
│ [Chat] →                       │
└────────────────────────────────┘
```

**New Features:**
- Quick access navigation cards
- Three buttons: Manage Jobs, All Applicants, Company Info
- Links to new pages from dashboard

---

## Color Palette

```
PRIMARY & BACKGROUNDS
├── Primary: #4F46E5 (Indigo)
├── Muted: #F1F5F9 (Light Gray)
├── Border: #E2E8F0 (Border Gray)
└── Background: #F8FAFC (Page BG)

STATUS COLORS
├── New:         #DBEAFE (bg) | #2563EB (text) 🔵
├── Shortlisted: #DCFCE7 (bg) | #15803D (text) ✅
├── Rejected:    #FEE2E2 (bg) | #DC2626 (text) ❌
├── Active:      #DCFCE7 (bg) | #15803D (text) ✅
└── Closed:      #FEE2E2 (bg) | #DC2626 (text) ❌

TEXT COLORS
├── Primary Text: #000000 (Black)
├── Secondary: #64748B (Dark Gray)
└── Muted: #94A3B8 (Light Gray)
```

---

## Responsive Design

### Mobile (< 360px)
- Single column layouts
- Full-width buttons
- Stacked cards
- Bottom sheet modals

### Tablet (≥ 600px)
- Two column layouts
- Side-by-side cards
- Grid arrangements
- Full-screen modals

### Desktop (≥ 1200px)
- Multi-column layouts
- Expanded detail views
- Sidebar navigation
- Optimized spacing

---

## Typography Hierarchy

```
Page Title:        20px, Weight 700 (Bold)
Section Title:     16px, Weight 700 (Bold)
Subtitle:          14px, Weight 500
Body Text:         14px, Weight 400
Small Text:        12px, Weight 500
Badge Text:        11px, Weight 600
```

---

## Spacing Grid

```
Base Unit: 4px

Small:     8px (2 units)
Medium:    12px (3 units)
Large:     16px (4 units)
XLarge:    24px (6 units)
```

---

## Component Reference

### Status Badge
```
Width: 60-80px
Height: 24px
Padding: 4px horizontal, 6px vertical
Border Radius: 8px
Font: 11px, Weight 600
```

### Card
```
Border Radius: 14-16px
Padding: 14-20px
Border: 1px solid #E2E8F0
Shadow: 0 2px 8px rgba(0,0,0,0.03)
```

### Button (Primary)
```
Height: 52px
Border Radius: 16px
Background: #4F46E5
Text Color: White
Font: 14px, Weight 600
```

### Button (Outlined)
```
Height: 52px
Border Radius: 16px
Border: 1px solid #E2E8F0
Text Color: #4F46E5
Font: 14px, Weight 600
```

---

## Navigation Hierarchy

```
RecruiterDashboardPage
├── Quick Action Cards
│   ├── ManageJobsPage
│   ├── ApplicantsListPage
│   │   └── ApplicantDetailPage
│   └── CompanyProfilePage
└── Posted Jobs Section
    └── ApplicantsPage (Legacy)
```

---

## Summary

✅ **4 Full-Featured Pages**
- Mobile-first responsive design
- Consistent UI style system
- Interactive components
- Proper error handling
- Empty states for all views

✅ **Dashboard Integration**
- Quick access cards
- Seamless navigation
- Consistent branding

✅ **Ready for Backend**
- Mock data structure defined
- API integration points identified
- Data flow documented

# JOB POSTING PERSISTENCE ISSUE - DIAGNOSIS REPORT

**Status**: AUDIT COMPLETE (NO CODE CHANGES MADE)  
**Date**: 16 January 2026  
**Symptom**: Posted jobs disappear after Flutter app restart; not retrieved from DB on reload.

---

## 1. FINDINGS

### 1.1 POST Flow (Job Posting)
| Endpoint | Source Code | Writes To | Role | Status |
|----------|-------------|-----------|------|--------|
| `POST /jobs` | `api_main.py:522-555` | `jobs` DB table + `ARTIFACTS.job_lookup` (in-memory) | Recruiter only | ✅ Working |

**Detail**: `post_job()` calls `crud_jobs.create_job(db, ...)` which writes to Supabase `jobs` table with fields: `id, employer_user_id, title, description, location, category, company, salary, created_at`. Also updates in-memory cache `ARTIFACTS.job_lookup[new_id]`.

### 1.2 GET Flow (Job Listing)
| Endpoint | Source Code | Reads From | Used By | Status |
|----------|-------------|-----------|---------|--------|
| `GET /jobs` | `api_main.py:423-477` | `jobs` DB table + `ARTIFACTS.job_lookup` | Job Seeker Home, Search | ✅ DB-aware |
| `GET /jobs/{id}` | `api_main.py:479-515` | `jobs` DB table + `ARTIFACTS.job_lookup` | Job Details | ✅ DB-aware |
| `GET /recommendations` | `api_main.py:641-666` | **ONLY** `ARTIFACTS.job_lookup` | Recommendations on Home | ⚠️ Cache-only |
| `GET /saved` | `api_main.py:625-630` | **ONLY** `ARTIFACTS.job_lookup` | Saved Jobs page | ⚠️ Cache-only |

### 1.3 Flutter Job Loading Flow
| Page | API Call | When Called | Provider Updated | Persists on Restart |
|------|----------|-------------|------------------|---------------------|
| Job Seeker Home | `loadJobs()` → `GET /jobs` | `initState` + pull-to-refresh | `jobs` list | ✅ YES (via DB) |
| Search | `loadJobs(query=...)` → `GET /jobs?query=` | User types | `jobs` list | ✅ YES (via DB) |
| Recruiter Dashboard | **NONE** - uses `jobs.postedJobs` | Hardcoded from Provider | In-memory only | ❌ NO - lost on restart |
| Recruiter Manage Jobs | **NONE** - uses `jobs.postedJobs` | Hardcoded from Provider | In-memory only | ❌ NO - lost on restart |
| Job Details | `getJobDetails()` → `GET /jobs/{id}` | Navigate to detail | Not in Provider | ✅ YES (via DB) |

**Key Observation**: Recruiter "posted jobs" UI does NOT call any GET endpoint; it reads only `JobProvider.postedJobs`, which is populated during `postJob()` and LOST on app restart.

### 1.4 Data Persistence On App Restart

**Scenario: Recruiter posts a job**
1. ✅ `POST /jobs` → Saves to `jobs` table in Supabase
2. ✅ `POST /jobs` → Updates `ARTIFACTS.job_lookup` (in-memory)
3. ✅ `postJob()` in Provider → Adds to `JobProvider.postedJobs` list

**Scenario: Flutter app restarts**
1. ✅ New `JobProvider` instantiated (fresh `postedJobs = []`)
2. ❌ `initState()` calls `loadJobs()` which populates `jobs` list from DB
3. ❌ NO call to populate `postedJobs` from DB
4. ❌ Recruiter dashboard sees **empty** `postedJobs` list
5. ✅ Job seeker home sees jobs (because `loadJobs()` fetches from DB)

### 1.5 Schema Alignment ✅ CORRECT
**Database `jobs` table** (Supabase PostgreSQL):
- `id` (PK, String)
- `employer_user_id` (FK → users.id)
- `title`, `description`, `location`, `category`, `company`, `salary`, `created_at`

**API Response (`JobOut` model)**:
- `job_id` ← maps to DB `id`
- `job_title` ← maps to DB `title`
- `company`, `location`, `category`, `salary` (exact match)
- `descriptions` ← maps to DB `description` (note: plural in API)

**Flutter Job model**:
- `jobId`, `jobTitle`, `company`, `location`, `category`, `salary`, `descriptions`
- Field names match API response keys exactly ✅

**Result**: No schema mismatch. Field alignment is correct end-to-end.

### 1.6 RLS Policies
No Row-Level Security (RLS) policies detected in Alembic migration `902138dc949d_init_schema.py`. All reads/writes use service-role-equivalent (FastAPI backend connection with `DATABASE_URL`). **No RLS blocking suspected.**

### 1.7 In-Memory Cache Issues
- `ARTIFACTS.job_lookup` populated **once at startup** from CSV + DB (line 113-146, `HybridArtifacts.load_and_train()`)
- Updated on-the-fly when jobs posted OR fetched (lines 427-432, 503, 642-669)
- **BUT**: Recommendations and saved jobs endpoints **ONLY** use cache, never query DB (lines 641-666, 625-630)
- New jobs posted after startup will appear in `/jobs` listing ✅ but NOT in `/recommendations` ❌ (needs retraining)

---

## 2. ROOT CAUSE(S) - RANKED BY CONFIDENCE

### 🔴 **ROOT CAUSE #1 (Confidence: 95%)**
**Recruiter "Manage Jobs" page never fetches from DB**

- File: `lib/pages/recruiter/manage_jobs_page.dart:19-20`
- File: `lib/pages/recruiter/recruiter_dashboard_page.dart:41,57,67,70`
- **Problem**: Page directly reads `jobs.postedJobs` from Provider state
- **Why it fails**: `JobProvider.postedJobs` is populated ONLY during `postJob()` call; no endpoint refreshes it from DB
- **Impact**: CRITICAL - Recruiter cannot see their posted jobs after app restart

### 🟡 **ROOT CAUSE #2 (Confidence: 75%)**
**`JobProvider.postJob()` doesn't sync with DB**

- File: `lib/providers/job_provider.dart:78-101`
- **Problem**: After `postJob()` succeeds, Provider adds job to local `postedJobs` list, BUT there's no method to LOAD `postedJobs` from DB
- **Why it matters**: On app restart, Provider has no way to reload recruiter's posted jobs
- **Impact**: MODERATE - Architectural gap; no persistence mechanism for recruiter context

### 🟡 **ROOT CAUSE #3 (Confidence: 60%)**
**GET `/recommendations` and GET `/saved` only read cache, not DB**

- File: `api_main.py:641-666` (recommendations)
- File: `api_main.py:625-630` (saved)
- **Problem**: Both endpoints only look at `ARTIFACTS.job_lookup`, never query `jobs` table or `saved_jobs` table
- **Why it fails**: If saved/recommended jobs are new (posted after startup), they won't appear because cache wasn't updated
- **Impact**: MODERATE - Affects recommendations and saved jobs features; symptoms similar to main issue

---

## 3. PROPOSED FIX PLAN

### **Strategy: DB-First Single Source of Truth**
- Supabase `jobs` table is the canonical source
- JobStreet CSV is seed data only (merged into DB on startup)
- In-memory cache (`ARTIFACTS.job_lookup`) is optional performance layer, NOT required for correctness
- All new recruiter-posted jobs MUST be readable from DB immediately

---

### **Phase 1: Add Endpoints to Sync Recruiter Context (MINIMAL)**

**1.1 Add new API endpoint: `GET /recruiter/jobs`**
- Query DB for `jobs` table filtered by `employer_user_id == current_user.id`
- Return list of jobs posted by recruiter
- Files to add:
  - `api_main.py` (new endpoint ~10 lines after line 555)
  - Update `PostJobRequest` handling if needed

**1.2 Add Provider method: `loadPostedJobs(userId)`**
- Call new `GET /recruiter/jobs` endpoint
- Populate `JobProvider.postedJobs` from DB response
- Files to update:
  - `lib/providers/job_provider.dart` (add method ~15 lines after line 101)

---

### **Phase 2: Update Flutter Pages to Fetch Recruiter Jobs (MINIMAL)**

**2.1 Recruiter Dashboard/Manage Jobs page initialization**
- On `initState`, call `JobProvider.loadPostedJobs(user_id)`
- Files to update:
  - `lib/pages/recruiter/recruiter_dashboard_page.dart` (add ~5 lines in build or initState)
  - `lib/pages/recruiter/manage_jobs_page.dart` (add ~5 lines in build or initState)

**2.2 Post Job success → Reload postedJobs**
- After `postJob()` succeeds, call `loadPostedJobs()`
- Files to update:
  - `lib/pages/recruiter/post_job_page.dart` (add ~2 lines after postJob call)

---

### **Phase 3: Fix Recommendations and Saved Jobs (OPTIONAL BUT RECOMMENDED)**

**3.1 Update `GET /recommendations` to include newly posted jobs**
- Query DB for recently posted jobs (posted since last model train)
- Append to recommendations (marked with `source: "recruiter_recent"` or similar)
- Files to update:
  - `api_main.py` (lines 641-666: modify to query DB jobs table)

**3.2 Update `GET /saved` to query `saved_jobs` table from DB**
- Replace cache-only logic with proper DB query
- Files to update:
  - `api_main.py` (lines 625-630: query `saved_jobs` table)
  - `crud/saved_jobs.py` (verify `list_saved_jobs()` returns jobs with proper fields)

---

### **Phase 4: Add "Source" Field for Tracing (OPTIONAL - FOR DEBUGGING)**

**4.1 Add `source` field to `JobOut` model**
- Values: `"jobstreet"` (from CSV), `"recruiter_posted"` (from DB), `"user_posted"` (if job seeker can post)
- Helps with logging and debugging persistence issues
- Files to update:
  - `api_main.py` (JobOut model, line 267)
  - Backend job assembly logic (~3 places)
  - Flutter Job model (make optional)

---

### **Summary of Changes**

| Phase | Files | Lines Est. | Risk | Priority |
|-------|-------|-----------|------|----------|
| 1.1 | `api_main.py` | +15 | LOW | CRITICAL |
| 1.2 | `lib/providers/job_provider.dart` | +15 | LOW | CRITICAL |
| 2.1 | `lib/pages/recruiter/*.dart` | +10 | LOW | CRITICAL |
| 2.2 | `lib/pages/recruiter/post_job_page.dart` | +2 | LOW | CRITICAL |
| 3.1 | `api_main.py` | +20 | MEDIUM | RECOMMENDED |
| 3.2 | `api_main.py`, `crud/saved_jobs.py` | +15 | MEDIUM | RECOMMENDED |
| 4.1 | Multiple | +10 | LOW | OPTIONAL |
| **TOTAL** | | **~87** | | |

---

## 4. EXACT FILES & FUNCTIONS TO CHANGE (LATER)

### **CRITICAL FIXES (Required)**

#### **File: `api_main.py`**
- **Line ~555**: After `@app.post("/jobs")` block, ADD new endpoint:
  - Function name: `get_recruiter_jobs()`
  - Decorator: `@app.get("/recruiter/jobs")`
  - Approx. lines: 8-12

- **Line ~267**: Update `class JobOut` (if adding source field):
  - Add field: `source: str = "jobstreet"`
  - Approx. lines: 1-3

#### **File: `lib/providers/job_provider.dart`**
- **Line ~102**: After `postJob()` method, ADD new method:
  - Function name: `loadPostedJobs(String userId)`
  - Approx. lines: 12-15
  - Logic: Call API `GET /recruiter/jobs`, populate `postedJobs` list

- **Line ~101**: Modify `postJob()` to call `loadPostedJobs()` after insert
  - Add line: `await loadPostedJobs(???)` — need to get user ID
  - OR: Return user ID from postJob response

#### **File: `lib/pages/recruiter/recruiter_dashboard_page.dart`**
- **Line ~10-30**: In `_RecruiterMainPageState.initState()` or `build()`, ADD:
  - Call `context.read<JobProvider>().loadPostedJobs(auth.session?.profile.id)`
  - Approx. lines: 2-3

#### **File: `lib/pages/recruiter/manage_jobs_page.dart`**
- **Line ~10-30**: Similar to dashboard, ADD initState or build call:
  - Call `context.read<JobProvider>().loadPostedJobs(user_id)`
  - Approx. lines: 2-3

#### **File: `lib/pages/recruiter/post_job_page.dart`**
- **Line ~120-150** (in `_submitJob()` or submit callback): After successful `postJob()`, ADD:
  - Call `jobs.loadPostedJobs(user_id)`
  - Approx. lines: 1-2

---

### **RECOMMENDED FIXES (Improves overall consistency)**

#### **File: `api_main.py`**
- **Line ~641** (`user_recommendations`): Modify to query DB for recent recruiter jobs
  - Add: Query `jobs` table for `created_at > last_train_time`
  - Approx. lines: 10-15 new

- **Line ~625** (`list_saved`): Replace with proper DB query
  - Change from: `[ARTIFACTS.job_lookup[jid] for jid in saved_ids ...]`
  - Change to: Query `saved_jobs` JOIN `jobs` to get full job details
  - Approx. lines: Modify 1-5 lines

#### **File: `crud/saved_jobs.py`**
- **New method**: `get_saved_jobs_with_details()`
  - Query `saved_jobs` JOIN `jobs` to return Job objects with all fields
  - Approx. lines: 8-10 new

---

### **OPTIONAL FIXES (For debugging & traceability)**

#### **File: `api_main.py`**
- **Line ~267**: Add `source` field to `JobOut` Pydantic model
  - `source: str = "jobstreet"`

- **Lines ~435-480** (in `list_jobs`): When building JobOut, set `source`:
  - For CSV jobs: `source="jobstreet"`
  - For DB recruiter jobs: `source="recruiter_posted"`

- **Line ~520** (in `post_job` response): Set `source="recruiter_posted"`

#### **File: `lib/models/job.dart`**
- **Line ~8**: Add optional field: `source: String?`
- Update `fromJson()` to parse it

---

## 5. VERIFICATION STEPS

### **Manual Testing (User-facing)**

1. **Test: Recruiter posts job and sees it immediately**
   - Start app, login as recruiter
   - Navigate to "Post Job" page
   - Fill form, submit
   - Expected: Job appears in "Manage Jobs" or dashboard
   - Verify: ✅ Job shows up (already working due to Provider.postJob)

2. **Test: Recruiter restarts app, job persists**
   - After step 1, force-close Flutter app (kill process)
   - Restart Flutter app
   - Login as recruiter
   - Navigate to "Manage Jobs" or dashboard
   - Expected: ✅ Previously posted job still visible
   - Current: ❌ Job is gone (demonstrates issue)

3. **Test: Job seeker sees recruiter's job**
   - In same session or new session, login as job seeker
   - Navigate to "Home" or "Browse Jobs"
   - Expected: ✅ Recruiter's job appears in list
   - Verify: ✅ Already works (GET /jobs queries DB)

4. **Test: Recommendations include recruiter's job**
   - Login as job seeker
   - Check "Recommended" section
   - Expected: ✅ New recruiter jobs may appear (if matched)
   - Current: ⚠️ May not appear until API restart (cache-only)

5. **Test: Search finds recruiter's job**
   - Search by job title keyword
   - Expected: ✅ Job found (GET /jobs filters correctly)
   - Verify: ✅ Already works

---

### **Automated Tests (To Add)**

#### **Test File: `test_recruiter_job_persistence.py`** (Python)
```
Test: recruiter posts job, API returns it in GET /recruiter/jobs
  1. POST /jobs as recruiter with test job data
  2. GET /recruiter/jobs (with auth)
  3. Assert test job in response
  4. Kill API + restart
  5. GET /recruiter/jobs again
  6. Assert test job still in response (from DB, not cache)
```

#### **Test File: `test_job_seeker_sees_recruiter_job.dart`** (Flutter)
```
Test: job seeker sees recruiter's newly posted job
  1. Post job as recruiter
  2. Logout, login as job seeker
  3. Call JobProvider.loadJobs()
  4. Assert job in jobs list
  5. Kill app + restart
  6. Call loadJobs() again
  7. Assert job still in list
```

---

### **Database Verification (Direct Supabase)**

```sql
-- Verify job was inserted
SELECT id, title, company, employer_user_id, created_at 
FROM jobs 
WHERE employer_user_id = '<recruiter_user_id>'
ORDER BY created_at DESC 
LIMIT 5;

-- Verify schema is correct
\d jobs;

-- Check for any null/missing fields
SELECT * FROM jobs 
WHERE title IS NULL OR description IS NULL 
LIMIT 5;
```

---

### **Logging to Add (For Debugging)**

- `POST /jobs`: Log `"Posted job {job_id} to DB, cache, and Provider"`
- `GET /recruiter/jobs`: Log `"Fetched {n} jobs from DB for recruiter {user_id}"`
- `Provider.loadPostedJobs()`: Log `"Loaded {n} posted jobs from API"`
- `Provider.postJob()`: Log `"Posted job {job_id}, calling loadPostedJobs()"`

---

## 6. REGRESSION PREVENTION

### **1. Add Smoke Test on App Startup**
- On app launch after login, call `jobs.loadJobs()` and verify >= 1 job in response
- Log result: `"✓ App startup: {n} jobs loaded from API"`
- If 0 jobs, log warning

### **2. Add Regression Test Suite**
- Run in CI/CD after each commit:
  - `test_job_persistence.py` (API + DB)
  - `test_flutter_job_listing.dart` (Flutter integration)

### **3. Monitor Production (Future)**
- Log: "jobs.postedJobs.length" on recruiter dashboard open
- Alert if suddenly drops to 0 on next session

---

## 7. FINAL RECOMMENDATION

**Implement ONLY Phase 1 + Phase 2 (CRITICAL FIXES)**
- Low risk, high impact
- ~30 LOC total
- Fixes root cause immediately
- No refactoring of unrelated code
- Estimated time: 15-20 minutes

**Phase 3 (Recommendations/Saved) can defer** — nice-to-have consistency fix

**Phase 4 (Source field) can defer** — nice-to-have debugging aid

---

## 8. IMPACT SUMMARY

| Issue | Before Fix | After Fix |
|-------|-----------|-----------|
| Recruiter posts job, app restarts, job visible? | ❌ NO | ✅ YES |
| Job seeker sees recruiter's job immediately? | ✅ YES (via GET /jobs) | ✅ YES (unchanged) |
| Recommendations include new recruiter jobs? | ⚠️ NO (cache-only) | ⚠️ NO (needs Phase 3 + retraining) |
| Saved jobs load correctly? | ⚠️ PARTIAL (cache-only) | ✅ YES (Phase 3 recommended) |
| Code complexity increase? | — | 🟢 Minimal (~30 LOC) |

---

## AUDIT COMPLETE
**Next**: Await approval to proceed with implementation.

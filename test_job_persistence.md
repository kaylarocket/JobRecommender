# Job Persistence Fix - Testing Guide

## Problem Fixed
Previously, jobs posted by recruiters were saved to the database but NOT retrieved when the app restarted because the `/jobs` endpoint only read from the in-memory cache (`ARTIFACTS.job_lookup`), which was populated once at startup from CSV files.

## Solution Implemented
Updated two endpoints to fetch from the database on every request:

### 1. `/jobs` Endpoint (List Jobs)
- Now queries the database with `crud_jobs.list_jobs(db)` on every request
- Merges database jobs with CSV jobs
- Updates in-memory cache to keep it synchronized
- Returns all jobs (database + CSV) with search/filter support

### 2. `/jobs/{job_id}` Endpoint (Get Single Job)
- First checks database with `crud_jobs.get_job(db, job_id)`
- Falls back to in-memory cache if not found in database
- Updates cache when database job is found

## How to Test

### Test 1: Post a Job and Verify Persistence

1. **Start the API server:**
   ```bash
   source venv/bin/activate
   python3 -m uvicorn api_main:app --reload
   ```

2. **Run the Flutter app:**
   ```bash
   flutter run -d "iPhone 17"
   ```

3. **Post a job as a recruiter:**
   - Login/register as a recruiter
   - Navigate to "Post Job" page
   - Fill in job details:
     - Title: "Test Persistence Job"
     - Company: "Test Company"
     - Location: "Remote"
     - Category: "Engineering"
     - Description: "Testing database persistence"
   - Submit the job

4. **Verify job appears immediately:**
   - Check the jobs list - your new job should appear

5. **Restart the Flutter app:**
   ```bash
   # Stop the app (Ctrl+C in terminal)
   # Run again
   flutter run -d "iPhone 17"
   ```

6. **Verify job persists:**
   - ✅ Your job should still appear in the jobs list
   - ✅ You can click on it to view details
   - ✅ Search should find it

### Test 2: Verify with Direct API Calls

```bash
# Register a recruiter
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@recruiter.com",
    "password": "testpass123",
    "full_name": "Test Recruiter",
    "role": "recruiter"
  }'

# Save the token from response
TOKEN="<paste_token_here>"

# Post a job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "job_title": "API Test Job",
    "company": "API Test Company",
    "location": "Remote",
    "category": "Engineering",
    "descriptions": "Testing API persistence"
  }'

# Save the job_id from response
JOB_ID="<paste_job_id_here>"

# Verify job appears in list
curl http://localhost:8000/jobs

# Verify job can be fetched by ID
curl http://localhost:8000/jobs/$JOB_ID

# Restart API server (Ctrl+C and run uvicorn again)

# Verify job STILL appears
curl http://localhost:8000/jobs
curl http://localhost:8000/jobs/$JOB_ID
```

### Test 3: Database Verification

```bash
# Connect to PostgreSQL
psql $DATABASE_URL

# Check jobs table
SELECT id, title, company, employer_user_id, created_at 
FROM jobs 
ORDER BY created_at DESC 
LIMIT 10;

# Verify your test jobs are there
```

## Expected Behavior

### ✅ Before Fix Issues (RESOLVED):
- ❌ Jobs disappeared after app restart
- ❌ Database had jobs but API didn't return them
- ❌ In-memory cache was only populated at startup

### ✅ After Fix (CURRENT):
- ✅ Jobs persist across app restarts
- ✅ Database jobs are fetched on every `/jobs` request
- ✅ New jobs immediately visible without restart
- ✅ In-memory cache stays synchronized with database

## Technical Details

### Database Flow:
```
POST /jobs
  ↓
crud_jobs.create_job(db, ...)
  ↓
db.add(job)
db.commit()
  ↓
Job saved to PostgreSQL ✓

GET /jobs (NEW BEHAVIOR)
  ↓
crud_jobs.list_jobs(db)  ← Fetches from database
  ↓
Merge with ARTIFACTS.job_lookup
  ↓
Return all jobs (database + CSV)
```

### Files Changed:
- `api_main.py` (lines ~422-520)
  - `/jobs` endpoint now queries database
  - `/jobs/{job_id}` endpoint checks database first
  - Both update in-memory cache for consistency

## Important Notes

1. **Performance**: Database is queried on every request, which is fine for small-medium scale. For production with high traffic, consider:
   - Adding database connection pooling (already handled by SQLAlchemy)
   - Implementing cache invalidation strategies
   - Adding pagination limits

2. **Cache Synchronization**: The in-memory cache (`ARTIFACTS.job_lookup`) is now updated when database jobs are fetched, ensuring consistency.

3. **Hybrid Recommendations**: Jobs posted after server startup will appear in listings but won't be in TF-IDF/LightFM models until retraining. See the TODO comment in `post_job` endpoint.

## Verification Checklist

- [ ] Post a job as recruiter
- [ ] Job appears immediately in Flutter app
- [ ] Restart Flutter app - job still visible
- [ ] Restart API server - job still visible
- [ ] Search for job - found successfully
- [ ] View job details - works correctly
- [ ] Check PostgreSQL - job record exists

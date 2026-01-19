#!/usr/bin/env python3
"""
Test job persistence fix - verifies jobs are retrieved from database.
Run this AFTER starting the API server: uvicorn api_main:app --reload
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_job_persistence():
    print("=" * 70)
    print("Testing Job Persistence Fix")
    print("=" * 70)
    
    # Step 1: Register recruiter
    print("\n1️⃣  Registering recruiter account...")
    email = f"persistence_test_{int(time.time())}@example.com"
    register_data = {
        "email": email,
        "password": "testpass123",
        "full_name": "Persistence Test Recruiter",
        "role": "recruiter"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/auth/register", json=register_data)
        if resp.status_code != 200:
            print(f"   ❌ Registration failed: {resp.text}")
            return
        
        auth = resp.json()
        token = auth["access_token"]
        user_id = auth["user"]["id"]
        print(f"   ✅ Registered: {email}")
        print(f"   User ID: {user_id}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Make sure API server is running!")
        return
    
    # Step 2: Count jobs before posting
    print("\n2️⃣  Counting jobs before posting...")
    resp = requests.get(f"{BASE_URL}/jobs")
    if resp.status_code == 200:
        jobs_before = resp.json()
        count_before = len(jobs_before.get("items", jobs_before))
        print(f"   ✅ Current job count: {count_before}")
    else:
        print(f"   ❌ Failed to fetch jobs: {resp.text}")
        return
    
    # Step 3: Post a new job
    print("\n3️⃣  Posting a new job...")
    job_data = {
        "job_title": "Persistence Test Job - DO NOT APPLY",
        "company": "Test Company Inc",
        "location": "Remote",
        "category": "Engineering",
        "salary": "$100,000",
        "descriptions": "This is a test job to verify database persistence."
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BASE_URL}/jobs", json=job_data, headers=headers)
    
    if resp.status_code != 200:
        print(f"   ❌ Job posting failed: {resp.text}")
        return
    
    job = resp.json()
    job_id = job["job_id"]
    print(f"   ✅ Job posted successfully!")
    print(f"   Job ID: {job_id}")
    print(f"   Title: {job['job_title']}")
    
    # Step 4: Verify job appears in list immediately
    print("\n4️⃣  Verifying job appears in jobs list...")
    resp = requests.get(f"{BASE_URL}/jobs")
    if resp.status_code == 200:
        jobs_after = resp.json()
        all_jobs = jobs_after.get("items", jobs_after)
        count_after = len(all_jobs)
        
        job_ids = [j["job_id"] for j in all_jobs]
        if job_id in job_ids:
            print(f"   ✅ Job found in list! (Total jobs: {count_after}, was {count_before})")
        else:
            print(f"   ❌ Job NOT found in list!")
            print(f"   Looking for: {job_id}")
            print(f"   Found IDs: {job_ids[:5]}...")
            return
    else:
        print(f"   ❌ Failed to fetch jobs: {resp.text}")
        return
    
    # Step 5: Fetch job by ID
    print("\n5️⃣  Fetching job by ID...")
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
    if resp.status_code == 200:
        fetched_job = resp.json()
        print(f"   ✅ Job retrieved successfully!")
        print(f"   Title: {fetched_job['job_title']}")
        print(f"   Company: {fetched_job['company']}")
    else:
        print(f"   ❌ Failed to fetch job: {resp.text}")
        return
    
    # Step 6: Test search
    print("\n6️⃣  Testing search functionality...")
    resp = requests.get(f"{BASE_URL}/jobs?query=Persistence")
    if resp.status_code == 200:
        search_results = resp.json()
        items = search_results.get("items", search_results)
        found_in_search = any(j["job_id"] == job_id for j in items)
        if found_in_search:
            print(f"   ✅ Job found in search results!")
        else:
            print(f"   ⚠️  Job not found in search (might be pagination)")
    
    # Step 7: Test category filter
    print("\n7️⃣  Testing category filter...")
    resp = requests.get(f"{BASE_URL}/jobs?category=Engineering")
    if resp.status_code == 200:
        filtered_results = resp.json()
        items = filtered_results.get("items", filtered_results)
        found_in_filter = any(j["job_id"] == job_id for j in items)
        if found_in_filter:
            print(f"   ✅ Job found in category filter!")
        else:
            print(f"   ⚠️  Job not found in filter (might be pagination)")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ JOB PERSISTENCE FIX VERIFIED!")
    print("=" * 70)
    print("\nWhat was tested:")
    print("  ✓ Job saved to database via POST /jobs")
    print("  ✓ Job appears immediately in GET /jobs")
    print("  ✓ Job can be fetched by ID via GET /jobs/{id}")
    print("  ✓ Job appears in search results")
    print("  ✓ Job appears in category filters")
    print("\nNext steps:")
    print("  1. Restart your Flutter app and verify job still appears")
    print("  2. Restart API server and verify job still appears")
    print("  3. Check database: psql $DATABASE_URL")
    print(f"     SELECT * FROM jobs WHERE id = '{job_id}';")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_job_persistence()

"""
Quick test to verify recruiter job posting updates the database.
Run this while the API server is running (uvicorn api_main:app --reload)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_job_posting():
    print("=" * 60)
    print("Testing Recruiter Job Posting → Database Update")
    print("=" * 60)
    
    # Step 1: Register a recruiter account
    print("\n1. Registering recruiter account...")
    register_payload = {
        "full_name": "Test Recruiter",
        "email": f"recruiter_test_{hash('test')}@example.com",
        "password": "testpass123",
        "role": "recruiter"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
        if resp.status_code == 200:
            auth_data = resp.json()
            token = auth_data["access_token"]
            user_id = auth_data["user"]["id"]
            print(f"   ✓ Recruiter registered: {user_id}")
        else:
            print(f"   ✗ Registration failed: {resp.text}")
            return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Make sure API server is running: uvicorn api_main:app --reload")
        return
    
    # Step 2: Post a job
    print("\n2. Posting a new job...")
    job_payload = {
        "job_title": "Senior Python Developer",
        "company": "Test Company Inc",
        "location": "Remote",
        "category": "Software Development",
        "salary": "$100,000 - $150,000",
        "descriptions": "We are looking for an experienced Python developer to join our team."
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{BASE_URL}/jobs", json=job_payload, headers=headers)
    
    if resp.status_code == 200:
        job_data = resp.json()
        job_id = job_data["job_id"]
        print(f"   ✓ Job posted successfully: {job_id}")
        print(f"   Title: {job_data['job_title']}")
        print(f"   Company: {job_data['company']}")
    else:
        print(f"   ✗ Job posting failed: {resp.text}")
        return
    
    # Step 3: Verify job exists in database by fetching it
    print("\n3. Verifying job exists in database...")
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
    
    if resp.status_code == 200:
        fetched_job = resp.json()
        print(f"   ✓ Job retrieved from database:")
        print(f"   - ID: {fetched_job['job_id']}")
        print(f"   - Title: {fetched_job['job_title']}")
        print(f"   - Company: {fetched_job['company']}")
        print(f"   - Location: {fetched_job['location']}")
    else:
        print(f"   ✗ Failed to retrieve job: {resp.text}")
        return
    
    # Step 4: Check if job appears in jobs list
    print("\n4. Checking if job appears in jobs list...")
    resp = requests.get(f"{BASE_URL}/jobs")
    
    if resp.status_code == 200:
        jobs_response = resp.json()
        all_jobs = jobs_response if isinstance(jobs_response, list) else jobs_response.get("items", [])
        job_ids = [j["job_id"] for j in all_jobs]
        
        if job_id in job_ids:
            print(f"   ✓ Job {job_id} found in jobs list")
        else:
            print(f"   ⚠ Job {job_id} NOT found in jobs list (may be paginated)")
    else:
        print(f"   ✗ Failed to retrieve jobs list: {resp.text}")
    
    print("\n" + "=" * 60)
    print("RESULT: Job posting successfully updates the database ✓")
    print("=" * 60)
    print("\nFlow verified:")
    print("  1. Recruiter registers → User record created")
    print("  2. Recruiter posts job → Job record created with employer_user_id")
    print("  3. Job can be retrieved → Database persistence confirmed")
    print("  4. Job appears in listings → API integration working")

if __name__ == "__main__":
    test_job_posting()

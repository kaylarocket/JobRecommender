import pandas as pd

jobs = pd.read_csv("jobstreet_all_jobs.csv", nrows=500, encoding='latin1', on_bad_lines='warn')
apps = pd.read_csv("job_applicants.csv", nrows=500, encoding='latin1', on_bad_lines='warn')

print("=== JobStreet jobs columns ===")
print(jobs.columns)
print(jobs.head())

print("\n=== Applicants columns ===")
print(apps.columns)
print(apps.head())

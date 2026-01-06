from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from algorithms.core.data_loading import (
    DEFAULT_JOBSTREET_CSV,
    JOB_CATEGORY_COL,
    JOB_DESC_COL,
    JOB_ID_COL,
    JOB_LOCATION_COL,
    JOB_TITLE_COL,
)
from crud.applications import create_application
from crud.jobs import create_job, get_job
from crud.users import create_user, get_user_by_email
from db.session import SessionLocal
from models.application import Application
from models.job import Job
from models.saved_job import SavedJob
from models.user import User

DATA_DIR = Path(__file__).resolve().parents[1] / "api_data"
USERS_FILE = DATA_DIR / "users.json"
APPLICATIONS_FILE = DATA_DIR / "applications.json"
SAVED_JOBS_FILE = DATA_DIR / "saved_jobs.json"
CUSTOM_JOBS_FILE = DATA_DIR / "custom_jobs.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _job_lookup_from_csv(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df[JOB_ID_COL] = df[JOB_ID_COL].astype(str)
    return {str(row[JOB_ID_COL]): row for row in df.to_dict(orient="records")}


def _ensure_job(db: Session, job_id: str, raw: dict) -> Job:
    job = get_job(db, job_id)
    if job:
        return job
    return create_job(
        db=db,
        job_id=job_id,
        title=str(raw.get(JOB_TITLE_COL, "")),
        description=str(raw.get(JOB_DESC_COL, "")),
        location=str(raw.get(JOB_LOCATION_COL, "")),
        category=str(raw.get(JOB_CATEGORY_COL, "")),
        company=str(raw.get("company", "")),
        salary=str(raw.get("salary", "")),
        employer_user_id=None,
        skills_text=None,
    )


def _seed_jobstreet_jobs(db: Session, job_lookup: Dict[str, dict]) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    existing_ids = {row[0] for row in db.execute(select(Job.id)).all()}
    for job_id, raw in job_lookup.items():
        if job_id in existing_ids:
            skipped += 1
            continue
        db.add(
            Job(
                id=job_id,
                employer_user_id=None,
                title=str(raw.get(JOB_TITLE_COL, "")),
                description=str(raw.get(JOB_DESC_COL, "")),
                location=str(raw.get(JOB_LOCATION_COL, "")),
                category=str(raw.get(JOB_CATEGORY_COL, "")),
                company=str(raw.get("company", "")),
                salary=str(raw.get("salary", "")),
            )
        )
        inserted += 1
    if inserted:
        db.commit()
    return inserted, skipped


def _seed_custom_jobs(db: Session, custom_jobs: Iterable[dict]) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    for raw in custom_jobs:
        job_id = str(raw.get(JOB_ID_COL) or raw.get("job_id") or "").strip()
        if not job_id:
            continue
        if get_job(db, job_id):
            skipped += 1
            continue
        create_job(
            db=db,
            job_id=job_id,
            title=str(raw.get(JOB_TITLE_COL, "")),
            description=str(raw.get(JOB_DESC_COL, "")),
            location=str(raw.get(JOB_LOCATION_COL, "")),
            category=str(raw.get(JOB_CATEGORY_COL, "")),
            company=str(raw.get("company", "")),
            salary=str(raw.get("salary", "")),
            employer_user_id=None,
            skills_text=None,
        )
        inserted += 1
    return inserted, skipped


def _migrate_users(db: Session, users_payload: dict) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    for user_id, raw in users_payload.items():
        if db.get(User, user_id):
            skipped += 1
            continue
        if get_user_by_email(db, raw.get("email", "")):
            skipped += 1
            continue
        user = create_user(
            db=db,
            user_id=user_id,
            email=raw.get("email", ""),
            password_hash=raw.get("hashed_password", ""),
            role=raw.get("role", "job_seeker"),
            full_name=raw.get("full_name"),
            location=raw.get("preferred_location"),
            headline=raw.get("headline"),
            skills_text=raw.get("skills"),
            experience_years=raw.get("experience_years"),
        )
        created_at = _parse_dt(raw.get("created_at"))
        if created_at:
            user.created_at = created_at
            db.commit()
        inserted += 1
    return inserted, skipped


def _migrate_applications(
    db: Session,
    applications_payload: list[dict],
    job_lookup: Dict[str, dict],
) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    for raw in applications_payload:
        app_id = str(raw.get("id", "")).strip()
        if not app_id:
            continue
        if db.get(Application, app_id):
            skipped += 1
            continue
        if db.get(User, raw.get("user_id")) is None:
            skipped += 1
            continue
        if db.get(Job, raw.get("job_id")) is None:
            lookup = job_lookup.get(str(raw.get("job_id")))
            if lookup:
                _ensure_job(db, str(raw.get("job_id")), lookup)
            else:
                skipped += 1
                continue
        application = create_application(
            db=db,
            application_id=app_id,
            user_id=raw.get("user_id"),
            job_id=raw.get("job_id"),
            status=raw.get("status", "submitted"),
            cover_letter=raw.get("cover_letter"),
        )
        created_at = _parse_dt(raw.get("created_at"))
        if created_at:
            application.created_at = created_at
            db.commit()
        inserted += 1
    return inserted, skipped


def _migrate_saved_jobs(
    db: Session,
    saved_payload: dict,
    job_lookup: Dict[str, dict],
) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    for user_id, jobs in saved_payload.items():
        if db.get(User, user_id) is None:
            skipped += len(jobs)
            continue
        for job_id in jobs:
            job_id = str(job_id)
            if db.get(Job, job_id) is None:
                lookup = job_lookup.get(job_id)
                if lookup:
                    _ensure_job(db, job_id, lookup)
                else:
                    skipped += 1
                    continue
            if db.get(SavedJob, {"user_id": user_id, "job_id": job_id}):
                skipped += 1
                continue
            db.add(SavedJob(user_id=user_id, job_id=job_id))
            inserted += 1
        db.commit()
    return inserted, skipped


def main() -> None:
    users_payload = _load_json(USERS_FILE, {})
    applications_payload = _load_json(APPLICATIONS_FILE, [])
    saved_payload = _load_json(SAVED_JOBS_FILE, {})
    custom_jobs_payload = _load_json(CUSTOM_JOBS_FILE, [])

    job_lookup = _job_lookup_from_csv(DEFAULT_JOBSTREET_CSV)

    with SessionLocal() as db:
        seed_inserted, seed_skipped = _seed_jobstreet_jobs(db, job_lookup)
        print(f"Seeded JobStreet jobs: inserted={seed_inserted}, skipped={seed_skipped}")

        custom_inserted, custom_skipped = _seed_custom_jobs(db, custom_jobs_payload)
        print(f"Custom jobs: inserted={custom_inserted}, skipped={custom_skipped}")

        user_inserted, user_skipped = _migrate_users(db, users_payload)
        print(f"Users: inserted={user_inserted}, skipped={user_skipped}")

        app_inserted, app_skipped = _migrate_applications(db, applications_payload, job_lookup)
        print(f"Applications: inserted={app_inserted}, skipped={app_skipped}")

        saved_inserted, saved_skipped = _migrate_saved_jobs(db, saved_payload, job_lookup)
        print(f"Saved jobs: inserted={saved_inserted}, skipped={saved_skipped}")


if __name__ == "__main__":
    main()

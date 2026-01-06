from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from algorithms.core.data_loading import JOB_DESC_COL, JOB_ID_COL, JOB_LOCATION_COL, JOB_TITLE_COL
from models.job import Job


def get_job(db: Session, job_id: str) -> Optional[Job]:
    return db.get(Job, job_id)


def create_job(
    db: Session,
    job_id: str,
    title: str,
    description: Optional[str],
    location: Optional[str],
    category: Optional[str],
    company: Optional[str],
    salary: Optional[str],
    employer_user_id: Optional[str] = None,
    skills_text: Optional[str] = None,
) -> Job:
    job = Job(
        id=job_id,
        title=title,
        description=description,
        location=location,
        category=category,
        company=company,
        salary=salary,
        employer_user_id=employer_user_id,
        skills_text=skills_text,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def list_jobs(db: Session) -> list[Job]:
    stmt = select(Job)
    return list(db.execute(stmt).scalars().all())


def ensure_job_from_lookup(db: Session, job_id: str, job_lookup: dict) -> Optional[Job]:
    job = get_job(db, job_id)
    if job:
        return job
    raw = job_lookup.get(job_id)
    if not raw:
        return None
    return create_job(
        db=db,
        job_id=job_id,
        title=str(raw.get(JOB_TITLE_COL, "")),
        description=str(raw.get(JOB_DESC_COL, "")),
        location=str(raw.get(JOB_LOCATION_COL, "")),
        category=str(raw.get("category", "")),
        company=str(raw.get("company", "")),
        salary=str(raw.get("salary", "")),
        employer_user_id=None,
        skills_text=None,
    )

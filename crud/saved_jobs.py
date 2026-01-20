from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.saved_job import SavedJob


def save_job(db: Session, user_id: str, job_id: str) -> list[str]:
    stmt = select(SavedJob).where(SavedJob.user_id == user_id, SavedJob.job_id == job_id)
    existing = db.execute(stmt).scalar_one_or_none()
    if not existing:
        db.add(SavedJob(user_id=user_id, job_id=job_id))
        db.commit()
    stmt_all = select(SavedJob.job_id).where(SavedJob.user_id == user_id)
    return list(db.execute(stmt_all).scalars().all())


def list_saved_jobs(db: Session, user_id: str) -> list[str]:
    stmt = select(SavedJob.job_id).where(SavedJob.user_id == user_id)
    return list(db.execute(stmt).scalars().all())


def remove_saved_job(db: Session, user_id: str, job_id: str) -> list[str]:
    stmt = select(SavedJob).where(SavedJob.user_id == user_id, SavedJob.job_id == job_id)
    existing = db.execute(stmt).scalar_one_or_none()
    if existing:
        db.delete(existing)
        db.commit()
    stmt_all = select(SavedJob.job_id).where(SavedJob.user_id == user_id)
    return list(db.execute(stmt_all).scalars().all())

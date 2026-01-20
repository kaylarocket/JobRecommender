from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from models.application import Application
from models.job import Job
from models.user import User


def create_application(
    db: Session,
    application_id: str,
    user_id: str,
    job_id: str,
    status: str,
    cover_letter: Optional[str],
) -> Application:
    application = Application(
        id=application_id,
        user_id=user_id,
        job_id=job_id,
        status=status,
        cover_letter=cover_letter,
    )
    db.add(application)
    db.commit()
    db.refresh(application)
    return application


def list_applications_by_user(db: Session, user_id: str) -> list[Application]:
    stmt = (
        select(Application)
        .where(Application.user_id == user_id)
        .order_by(Application.created_at.desc())
        .options(
            selectinload(Application.job),
            selectinload(Application.user).selectinload(User.profile),
        )
    )
    return list(db.execute(stmt).scalars().all())


def list_applications_all(db: Session) -> list[Application]:
    stmt = select(Application).order_by(Application.created_at.desc()).options(
        selectinload(Application.job),
        selectinload(Application.user).selectinload(User.profile),
    )
    return list(db.execute(stmt).scalars().all())


def list_applications_for_employer(db: Session, employer_user_id: str) -> list[Application]:
    stmt = (
        select(Application)
        .join(Job, Job.id == Application.job_id)
        .where(Job.employer_user_id == employer_user_id)
        .order_by(Application.created_at.desc())
        .options(
            selectinload(Application.job),
            selectinload(Application.user).selectinload(User.profile),
        )
    )
    return list(db.execute(stmt).scalars().all())


def get_application_with_job(db: Session, application_id: str) -> Application | None:
    stmt = (
        select(Application)
        .where(Application.id == application_id)
        .options(selectinload(Application.job))
    )
    return db.execute(stmt).scalar_one_or_none()


def update_application_status(
    db: Session,
    application_id: str,
    status: str,
) -> Application | None:
    application = db.get(Application, application_id)
    if not application:
        return None
    application.status = status
    db.commit()
    db.refresh(application)
    return application

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from models.application import Application


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
    stmt = select(Application).where(Application.user_id == user_id)
    return list(db.execute(stmt).scalars().all())


def list_applications_all(db: Session) -> list[Application]:
    stmt = select(Application)
    return list(db.execute(stmt).scalars().all())

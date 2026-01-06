from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from models.employer import Employer
from models.user import User
from models.user_profile import UserProfile


def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
    return db.get(User, user_id)


def get_user_with_profile(db: Session, user_id: str) -> Optional[User]:
    stmt = (
        select(User)
        .options(selectinload(User.profile), selectinload(User.employer))
        .where(User.id == user_id)
    )
    return db.execute(stmt).scalar_one_or_none()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    stmt = (
        select(User)
        .options(selectinload(User.profile), selectinload(User.employer))
        .where(User.email == email)
    )
    return db.execute(stmt).scalar_one_or_none()


def create_user(
    db: Session,
    user_id: str,
    email: str,
    password_hash: str,
    role: str,
    full_name: Optional[str],
    location: Optional[str],
    headline: Optional[str],
    skills_text: Optional[str],
    experience_years: Optional[int],
) -> User:
    user = User(
        id=user_id,
        email=email,
        password_hash=password_hash,
        role=role,
    )
    profile = UserProfile(
        user_id=user_id,
        full_name=full_name,
        location=location,
        skills_text=skills_text,
        desired_roles_text=None,
        education_text=None,
        headline=headline,
        experience_years=experience_years,
    )
    user.profile = profile

    if role == "recruiter":
        user.employer = Employer(user_id=user_id)

    db.add(user)
    db.commit()
    db.refresh(user)
    return user

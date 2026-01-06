from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    employer = relationship("Employer", back_populates="user", uselist=False, cascade="all, delete-orphan")
    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="employer")
    saved_jobs: Mapped[List["SavedJob"]] = relationship("SavedJob", back_populates="user", cascade="all, delete-orphan")
    applications: Mapped[List["Application"]] = relationship("Application", back_populates="user", cascade="all, delete-orphan")

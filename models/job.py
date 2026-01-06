from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    employer_user_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    skills_text: Mapped[Optional[str]] = mapped_column(Text)
    location: Mapped[Optional[str]] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(255))
    company: Mapped[Optional[str]] = mapped_column(String(255))
    salary: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    employer = relationship("User", back_populates="jobs")
    saved_jobs: Mapped[List["SavedJob"]] = relationship("SavedJob", back_populates="job", cascade="all, delete-orphan")
    applications: Mapped[List["Application"]] = relationship("Application", back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_jobs_location", "location"),
    )

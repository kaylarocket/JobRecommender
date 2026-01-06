from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class SavedJob(Base):
    __tablename__ = "saved_jobs"

    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="saved_jobs")
    job = relationship("Job", back_populates="saved_jobs")

    __table_args__ = (
        UniqueConstraint("user_id", "job_id", name="uq_saved_jobs_user_job"),
        Index("ix_saved_jobs_user", "user_id"),
        Index("ix_saved_jobs_job", "job_id"),
    )

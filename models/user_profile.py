from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    location: Mapped[Optional[str]] = mapped_column(String(255))
    skills_text: Mapped[Optional[str]] = mapped_column(String(2000))
    desired_roles_text: Mapped[Optional[str]] = mapped_column(String(1000))
    education_text: Mapped[Optional[str]] = mapped_column(String(1000))
    headline: Mapped[Optional[str]] = mapped_column(String(255))
    experience_years: Mapped[Optional[int]] = mapped_column(Integer)

    user = relationship("User", back_populates="profile")

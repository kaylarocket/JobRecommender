from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


class Employer(Base):
    __tablename__ = "employers"

    user_id: Mapped[str] = mapped_column(String(32), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(255))
    company_location: Mapped[Optional[str]] = mapped_column(String(255))

    user = relationship("User", back_populates="employer")

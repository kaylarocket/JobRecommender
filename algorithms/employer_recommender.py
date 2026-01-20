"""
Employer-side content-based recommender (job -> candidates).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from algorithms.models.tfidf_model import build_tfidf_representations
from models.application import Application
from models.job import Job
from models.user import User
from models.user_profile import UserProfile


def _join_text(parts: Iterable[Optional[str]]) -> str:
    tokens = [str(part).strip() for part in parts if part and str(part).strip()]
    return " ".join(tokens)


def _normalize_location(value: Optional[str]) -> str:
    return str(value).strip().lower() if value else ""


def _load_candidates(db_session: Session) -> List[tuple[User, UserProfile]]:
    stmt = (
        select(User, UserProfile)
        .join(UserProfile, UserProfile.user_id == User.id)
        .where(User.role == "job_seeker")
    )
    return list(db_session.execute(stmt).all())


def _load_applicants_for_job(
    db_session: Session,
    job_id: str,
) -> List[tuple[Application, User, Optional[UserProfile]]]:
    stmt = (
        select(Application, User, UserProfile)
        .join(User, User.id == Application.user_id)
        .outerjoin(UserProfile, UserProfile.user_id == User.id)
        .where(Application.job_id == job_id)
        .where(User.role == "job_seeker")
    )
    return list(db_session.execute(stmt).all())


def recommend_candidates_for_job(
    job_id: str | int,
    db_session: Session,
    top_n: int = 10,
    location_boost: float = 0.05,
) -> List[dict]:
    """
    Recommend candidates for a job using TF-IDF + cosine similarity.

    The algorithm builds a job text vector (title + description + skills) and
    a candidate text vector (skills + desired roles + education), fits TF-IDF
    on the combined corpus, and ranks candidates by cosine similarity (0-1).
    """
    if top_n <= 0:
        return []

    job_key = str(job_id)
    job = db_session.get(Job, job_key)
    if not job:
        raise ValueError(f"Job {job_id} not found.")

    candidates = _load_candidates(db_session)
    if not candidates:
        return []

    job_text = _join_text([job.title, job.description, job.skills_text])
    job_location = _normalize_location(job.location)

    candidate_rows: List[dict] = []
    for user, profile in candidates:
        candidate_rows.append(
            {
                "user_id": user.id,
                "user_text": _join_text(
                    [
                        profile.skills_text,
                        profile.desired_roles_text,
                        profile.education_text,
                    ]
                ),
                "location": profile.location or "",
            }
        )

    corpus = [job_text] + [row["user_text"] for row in candidate_rows]
    if not any(text.strip() for text in corpus):
        return []

    users_df = pd.DataFrame(candidate_rows)
    jobs_df = pd.DataFrame([{"job_text": job_text}])
    _, job_tfidf, user_tfidf = build_tfidf_representations(users_df, jobs_df)
    scores = cosine_similarity(job_tfidf, user_tfidf).ravel()

    if location_boost > 0 and job_location:
        locations = (
            users_df["location"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        match_mask = np.array(
            [loc and (job_location in loc or loc in job_location) for loc in locations],
            dtype=bool,
        )
        if match_mask.any():
            scores[match_mask] = np.minimum(scores[match_mask] * (1 + location_boost), 1.0)

    top_n = min(top_n, len(scores))
    order = np.argsort(-scores, kind="mergesort")[:top_n]

    return [
        {"user_id": users_df.loc[idx, "user_id"], "score": float(scores[idx])}
        for idx in order
    ]


def recommend_applicants_for_job(
    job_id: str | int,
    db_session: Session,
    top_n: int = 10,
    location_boost: float = 0.05,
) -> List[dict]:
    """
    Rank applicants who already applied for a job using TF-IDF + cosine similarity.

    This only considers users who submitted an application for the given job_id.
    """
    if top_n <= 0:
        return []

    job_key = str(job_id)
    job = db_session.get(Job, job_key)
    if not job:
        raise ValueError(f"Job {job_id} not found.")

    rows = _load_applicants_for_job(db_session, job_key)
    if not rows:
        return []

    job_text = _join_text([job.title, job.description, job.skills_text])
    job_location = _normalize_location(job.location)

    candidate_rows: List[dict] = []
    for application, user, profile in rows:
        full_name = profile.full_name if profile and profile.full_name else ""
        headline = profile.headline if profile else ""
        location = profile.location if profile else ""
        user_text = _join_text(
            [
                profile.skills_text if profile else "",
                profile.desired_roles_text if profile else "",
                profile.education_text if profile else "",
                headline,
            ]
        )
        candidate_rows.append(
            {
                "user_id": user.id,
                "full_name": full_name,
                "headline": headline,
                "location": location,
                "application_id": application.id,
                "status": application.status,
                "applied_at": application.created_at,
                "user_text": user_text,
            }
        )

    if not candidate_rows:
        return []

    output_keys = [
        "user_id",
        "full_name",
        "headline",
        "location",
        "application_id",
        "status",
        "applied_at",
    ]

    corpus = [job_text] + [row["user_text"] for row in candidate_rows]
    if not any(text.strip() for text in corpus):
        ordered = sorted(
            candidate_rows,
            key=lambda row: row["applied_at"] or datetime.min,
            reverse=True,
        )
        return [
            {**{key: row[key] for key in output_keys}, "score": 0.0}
            for row in ordered[: min(top_n, len(ordered))]
        ]

    users_df = pd.DataFrame(candidate_rows)
    jobs_df = pd.DataFrame([{"job_text": job_text}])
    _, job_tfidf, user_tfidf = build_tfidf_representations(users_df, jobs_df)
    scores = cosine_similarity(job_tfidf, user_tfidf).ravel()

    if location_boost > 0 and job_location:
        locations = (
            users_df["location"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        match_mask = np.array(
            [loc and (job_location in loc or loc in job_location) for loc in locations],
            dtype=bool,
        )
        if match_mask.any():
            scores[match_mask] = np.minimum(scores[match_mask] * (1 + location_boost), 1.0)

    top_n = min(top_n, len(scores))
    order = np.argsort(-scores, kind="mergesort")[:top_n]

    return [
        {**{key: candidate_rows[idx][key] for key in output_keys}, "score": float(scores[idx])}
        for idx in order
    ]


__all__ = ["recommend_candidates_for_job", "recommend_applicants_for_job"]

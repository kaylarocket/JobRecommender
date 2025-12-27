"""
Data loading, cleaning, and synthetic interaction generation for the hybrid job recommender.
"""

from __future__ import annotations

from typing import Iterable
from pathlib import Path

import numpy as np
import pandas as pd



# Default data locations (relative to algorithms/data/)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_JOBSTREET_CSV = DATA_DIR / "jobstreet_all_jobs.csv"
DEFAULT_APPLICANTS_CSV = DATA_DIR / "job_applicants.csv"


# Column configuration
JOB_ID_COL = "job_id"
JOB_TITLE_COL = "job_title"
JOB_DESC_COL = "descriptions"
JOB_LOCATION_COL = "location"
JOB_CATEGORY_COL = "category"

USER_ID_COL = "Unnamed: 0"
USER_SKILLS_COL = "HaveWorkedWith"
USER_DEGREE_COL = "EdLevel"
USER_PREFERRED_LOC_COL = "Country"
USER_TARGET_ROLE_COL = "MainBranch"
USER_YEARS_PRO_COL = "YearsCodePro"
USER_SKILL_LEVEL_COL = "ComputerSkills"

# Dataset limits for faster local experimentation
MAX_USERS = 2000
MAX_JOBS = 5000


def load_jobstreet_job_dataset(path: str | Path = DEFAULT_JOBSTREET_CSV) -> pd.DataFrame:
    """
    Load the JobStreet jobs CSV.

    Parameters
    ----------
    path : str
        CSV path. Defaults to DEFAULT_JOBSTREET_CSV.
    """
    print(f"Loading JobStreet jobs from: {path}")
    df = pd.read_csv(str(path))
    print("JobStreet jobs – first 5 rows:")
    print(df.head())
    return df


def load_applicants_dataset(path: str | Path = DEFAULT_APPLICANTS_CSV) -> pd.DataFrame:
    """
    Load the applicants CSV.

    Parameters
    ----------
    path : str
        CSV path. Defaults to DEFAULT_APPLICANTS_CSV.
    """
    print(f"Loading applicants from: {path}")
    df = pd.read_csv(str(path))
    print("Applicants – first 5 rows:")
    print(df.head())
    return df


def build_job_table(
    raw_jobs: pd.DataFrame,
    job_id_col: str = JOB_ID_COL,
    job_title_col: str = JOB_TITLE_COL,
    job_desc_col: str = JOB_DESC_COL,
    job_location_col: str = JOB_LOCATION_COL,
    job_category_col: str = JOB_CATEGORY_COL,
) -> pd.DataFrame:
    """
    Clean and engineer job features for downstream models.
    """
    df = raw_jobs.copy()
    for col in [job_id_col, job_title_col, job_desc_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' — check configuration.")

    df["job_id"] = df[job_id_col].astype(str)
    df["job_location"] = df.get(job_location_col, "").fillna("").astype(str)
    df["job_category"] = df.get(job_category_col, "").fillna("").astype(str)

    df["job_text"] = (
        df[job_title_col].fillna("").astype(str)
        + " "
        + df[job_desc_col].fillna("").astype(str)
        + " "
        + df["job_category"].fillna("").astype(str)
    ).str.lower()

    return df[["job_id", "job_text", "job_location", "job_category"]]


def build_user_table(
    raw_users: pd.DataFrame,
    user_id_col: str = USER_ID_COL,
    user_skills_col: str = USER_SKILLS_COL,
    user_degree_col: str = USER_DEGREE_COL,
    user_preferred_loc_col: str = USER_PREFERRED_LOC_COL,
    user_target_role_col: str = USER_TARGET_ROLE_COL,
    user_years_pro_col: str = USER_YEARS_PRO_COL,
    user_skill_level_col: str = USER_SKILL_LEVEL_COL,
) -> pd.DataFrame:
    """
    Clean and engineer user features for downstream models.
    """
    df = raw_users.copy()
    for col in [user_id_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' — check configuration.")

    df["user_id"] = df[user_id_col].astype(str)
    df["skills_text"] = df.get(user_skills_col, "").fillna("").astype(str)
    df["degree_text"] = df.get(user_degree_col, "").fillna("").astype(str)
    df["target_role"] = df.get(user_target_role_col, "").fillna("").astype(str)
    df["preferred_location"] = df.get(user_preferred_loc_col, "").fillna("").astype(str)

    df["skill_level_token"] = "skilllvl_" + df.get(user_skill_level_col, "").fillna("").astype(str)

    df["user_text"] = (
        df["skills_text"]
        + " "
        + df["degree_text"]
        + " "
        + df["target_role"]
        + " "
        + df["skill_level_token"]
    ).str.lower()

    return df[["user_id", "user_text", "preferred_location", "target_role"]].drop_duplicates("user_id")


def _tokenize(text: str, max_tokens: int = 10) -> list[str]:
    tokens = [t.strip() for t in text.split() if len(t) > 2]
    return tokens[:max_tokens]


def build_synthetic_interactions(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    max_positive: int = 10,
    seed: int = 42,
    location_bias: float = 0.7,
) -> pd.DataFrame:
    """
    Build synthetic implicit feedback because real interaction logs are absent.

    For each user:
      - tokenize user_text (skills + degree + role + exp + skill level)
      - find jobs whose job_text matches some of these tokens
      - bias towards jobs in the same Country
      - sample up to `max_positive` interactions per user
    """
    rng = np.random.default_rng(seed=seed)
    interactions = []

    jobs_local = jobs.copy()
    jobs_local["job_text_lower"] = jobs_local["job_text"].str.lower()
    jobs_local["job_location_lower"] = jobs_local["job_location"].str.lower()

    for _, user in users.iterrows():
        user_id = user["user_id"]
        pref_loc = str(user.get("preferred_location", "")).lower()
        user_text = str(user.get("user_text", "")).lower()

        tokens = _tokenize(user_text, max_tokens=10)
        candidates = jobs_local

        if tokens:
            mask_series = candidates["job_text_lower"].str.contains(tokens[0], regex=False)
            for token in tokens[1:]:
                mask_series |= candidates["job_text_lower"].str.contains(token, regex=False)
            candidates = candidates[mask_series]

        if candidates.empty:
            candidates = jobs_local

        if pref_loc:
            in_loc = candidates[candidates["job_location_lower"].str.contains(pref_loc, case=False, regex=False, na=False)]
            if not in_loc.empty and rng.random() < location_bias:
                candidates = in_loc

        n_pos = min(max_positive, len(candidates))
        sampled = candidates.sample(n=n_pos, random_state=None)

        for _, job_row in sampled.iterrows():
            interactions.append({"user_id": user_id, "job_id": job_row["job_id"], "weight": 1.0})

    interactions_df = pd.DataFrame(interactions)
    print(f"Synthetic interactions: {len(interactions_df)} rows")
    if not interactions_df.empty:
        print("Example interactions per user:")
        print(interactions_df["user_id"].value_counts().head())
    return interactions_df

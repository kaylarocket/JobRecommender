"""
FastAPI wrapper for the hybrid JobStreet recommender.

README (quick start)
- Install deps: pip install fastapi uvicorn[standard] pandas scikit-learn lightfm python-jose[cryptography]
- Run training+API: uvicorn api_main:app --reload
- TF-IDF is used for content similarity (job title/description/category vs user profile text).
- LightFM is used for collaborative filtering on synthetic interactions.
- Scores are normalized and blended in compute_hybrid_scores to serve recommendations to the Flutter app.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field

from data_loading import (
    MAX_JOBS,
    MAX_USERS,
    JOB_DESC_COL,
    JOB_ID_COL,
    JOB_LOCATION_COL,
    JOB_TITLE_COL,
    build_job_table,
    build_synthetic_interactions,
    build_user_table,
    load_applicants_dataset,
    load_jobstreet_job_dataset,
)
from models import (
    build_lightfm_dataset,
    build_tfidf_representations,
    compute_hybrid_scores,
    predict_lightfm_scores_for_user,
)


# ----------------------
# Auth / persistence
# ----------------------
DATA_DIR = Path("api_data")
USERS_FILE = DATA_DIR / "users.json"
APPLICATIONS_FILE = DATA_DIR / "applications.json"
SAVED_JOBS_FILE = DATA_DIR / "saved_jobs.json"
CUSTOM_JOBS_FILE = DATA_DIR / "custom_jobs.json"

SECRET_KEY = "dev-secret-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


def ensure_storage():
    DATA_DIR.mkdir(exist_ok=True)
    for file_path, default in [
        (USERS_FILE, {}),
        (APPLICATIONS_FILE, []),
        (SAVED_JOBS_FILE, {}),
        (CUSTOM_JOBS_FILE, []),
    ]:
        if not file_path.exists():
            file_path.write_text(json.dumps(default, indent=2))


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2))


def hash_password(password: str) -> str:
    salt = secrets.token_hex(8)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    if "$" not in stored:
        return False
    salt, digest = stored.split("$", 1)
    return hashlib.sha256((salt + password).encode()).hexdigest() == digest


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ----------------------
# Data prep + model init
# ----------------------
class HybridArtifacts:
    def __init__(self):
        self.jobs_raw = pd.DataFrame()
        self.jobs_features = pd.DataFrame()
        self.users_features = pd.DataFrame()
        self.vectorizer = None
        self.job_tfidf = None
        self.user_tfidf = None
        self.dataset = None
        self.model = None
        self.user_features_matrix = None
        self.item_features_matrix = None
        self.job_lookup: Dict[str, dict] = {}

    def load_and_train(self) -> None:
        """
        Load CSVs, engineer features, train TF-IDF and LightFM.
        TF-IDF: content-based similarity (job_text vs user_text).
        LightFM: collaborative filtering on synthetic interactions.
        Hybrid: compute_hybrid_scores blends normalized TF-IDF + LightFM.
        """
        raw_jobs = load_jobstreet_job_dataset()
        raw_users = load_applicants_dataset()

        if len(raw_jobs) > MAX_JOBS:
            raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
        if len(raw_users) > MAX_USERS:
            raw_users = raw_users.sample(MAX_USERS, random_state=42)

        raw_jobs[JOB_ID_COL] = raw_jobs[JOB_ID_COL].astype(str)
        custom_jobs = load_json(CUSTOM_JOBS_FILE, [])
        if custom_jobs:
            raw_jobs = pd.concat([raw_jobs, pd.DataFrame(custom_jobs)], ignore_index=True)
        self.job_lookup = {str(row[JOB_ID_COL]): row.to_dict() for _, row in raw_jobs.iterrows()}

        self.jobs_features = build_job_table(raw_jobs).reset_index(drop=True)
        self.users_features = build_user_table(raw_users).reset_index(drop=True)

        interactions_df = build_synthetic_interactions(self.users_features, self.jobs_features)

        (
            self.vectorizer,
            self.job_tfidf,
            self.user_tfidf,
        ) = build_tfidf_representations(self.users_features, self.jobs_features)

        (
            self.dataset,
            interactions,
            weights,
            self.user_features_matrix,
            self.item_features_matrix,
        ) = build_lightfm_dataset(
            interactions_df=interactions_df,
            users=self.users_features,
            jobs=self.jobs_features,
        )

        from models import train_lightfm  # local import to avoid cycle

        self.model = train_lightfm(
            interactions=interactions,
            weights=weights,
            user_features=self.user_features_matrix,
            item_features=self.item_features_matrix,
        )

    def nearest_user_id(self, user_text_vector) -> Optional[str]:
        if self.user_tfidf is None or self.users_features.empty:
            return None
        sims = (user_text_vector @ self.user_tfidf.T).toarray().ravel()
        if sims.max() <= 0:
            return None
        best_idx = int(np.argmax(sims))
        return self.users_features.loc[best_idx, "user_id"]

    def recommend(self, user_id: str, user_text: str, top_k: int = 10, alpha: float = 0.6):
        if self.vectorizer is None or self.dataset is None or self.model is None:
            raise HTTPException(status_code=500, detail="Models not initialized yet.")

        user_vec = self.vectorizer.transform([user_text.lower()])
        content_scores = (user_vec @ self.job_tfidf.T).toarray().ravel()

        proxy_user_id = user_id
        lfm_scores: np.ndarray
        user_id_map, _, _, _ = self.dataset.mapping()
        if proxy_user_id not in user_id_map:
            proxy_user_id = self.nearest_user_id(user_vec) or proxy_user_id

        if proxy_user_id in user_id_map:
            lfm_scores = predict_lightfm_scores_for_user(
                user_id=proxy_user_id,
                model=self.model,
                dataset=self.dataset,
                jobs=self.jobs_features,
                user_features=self.user_features_matrix,
                item_features=self.item_features_matrix,
            )
        else:
            lfm_scores = np.zeros(len(self.jobs_features))

        hybrid_scores, content_norm, lfm_norm = compute_hybrid_scores(
            content_scores=content_scores,
            lfm_scores=lfm_scores,
            alpha=alpha,
        )

        rec_df = self.jobs_features.copy()
        rec_df["content_score"] = content_norm
        rec_df["lfm_score"] = lfm_norm
        rec_df["final_score"] = hybrid_scores
        rec_df = rec_df.sort_values("final_score", ascending=False).head(top_k)
        return rec_df.reset_index(drop=True)


ARTIFACTS = HybridArtifacts()
ensure_storage()
ARTIFACTS.load_and_train()

USERS_DB: Dict[str, dict] = load_json(USERS_FILE, {})
APPLICATIONS_DB: List[dict] = load_json(APPLICATIONS_FILE, [])
SAVED_JOBS_DB: Dict[str, List[str]] = load_json(SAVED_JOBS_FILE, {})


# ----------------------
# Schemas
# ----------------------
class UserProfile(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: str
    preferred_location: Optional[str] = None
    headline: Optional[str] = None
    skills: Optional[str] = None
    experience_years: Optional[int] = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    full_name: str
    role: str = Field(pattern="^(job_seeker|recruiter)$")
    preferred_location: Optional[str] = None
    headline: Optional[str] = None
    skills: Optional[str] = None
    experience_years: Optional[int] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserProfile


class JobOut(BaseModel):
    job_id: str
    job_title: str
    company: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    salary: Optional[str] = None
    descriptions: Optional[str] = None


class RecommendationOut(BaseModel):
    job_id: str
    job_title: str
    company: Optional[str] = None
    location: Optional[str] = None
    category: Optional[str] = None
    salary: Optional[str] = None
    final_score: float
    content_score: float
    lfm_score: float


class JobListResponse(BaseModel):
    items: List[JobOut]
    page: int
    page_size: int
    total: int


class PostJobRequest(BaseModel):
    job_title: str
    company: str
    location: str
    category: str
    salary: Optional[str] = None
    descriptions: str


class ApplyRequest(BaseModel):
    job_id: str
    cover_letter: Optional[str] = None


# ----------------------
# FastAPI setup
# ----------------------
app = FastAPI(title="Hybrid Job Recommender API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Helpers
# ----------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None or user_id not in USERS_DB:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return USERS_DB[user_id]


def serialize_user(user: dict) -> UserProfile:
    return UserProfile(
        id=user["id"],
        email=user["email"],
        full_name=user.get("full_name", ""),
        role=user.get("role", "job_seeker"),
        preferred_location=user.get("preferred_location"),
        headline=user.get("headline"),
        skills=user.get("skills"),
        experience_years=user.get("experience_years"),
    )


def user_profile_text(user: dict) -> str:
    tokens = [
        user.get("full_name", ""),
        user.get("role", ""),
        user.get("preferred_location", ""),
        user.get("headline", ""),
        user.get("skills", ""),
        str(user.get("experience_years", "")),
    ]
    return " ".join([t for t in tokens if t]).lower()


def paginated_jobs(page: int, page_size: int) -> List[dict]:
    jobs = list(ARTIFACTS.job_lookup.values())
    start = (page - 1) * page_size
    end = start + page_size
    return jobs[start:end]


# ----------------------
# Auth endpoints
# ----------------------
@app.post("/auth/register", response_model=TokenResponse)
def register_user(payload: RegisterRequest):
    if any(u.get("email") == payload.email for u in USERS_DB.values()):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = secrets.token_hex(8)
    user_record = {
        "id": user_id,
        "email": payload.email,
        "full_name": payload.full_name,
        "role": payload.role,
        "preferred_location": payload.preferred_location,
        "headline": payload.headline,
        "skills": payload.skills,
        "experience_years": payload.experience_years,
        "hashed_password": hash_password(payload.password),
        "created_at": datetime.utcnow().isoformat(),
    }
    USERS_DB[user_id] = user_record
    save_json(USERS_FILE, USERS_DB)

    access_token = create_access_token({"sub": user_id})
    return TokenResponse(access_token=access_token, token_type="bearer", user=serialize_user(user_record))


@app.post("/auth/login", response_model=TokenResponse)
def login_user(payload: LoginRequest):
    user = next((u for u in USERS_DB.values() if u.get("email") == payload.email), None)
    if not user or not verify_password(payload.password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"sub": user["id"]})
    return TokenResponse(access_token=access_token, token_type="bearer", user=serialize_user(user))


# ----------------------
# Job endpoints
# ----------------------
@app.get("/jobs", response_model=JobListResponse)
def list_jobs(page: int = 1, page_size: int = 20, query: Optional[str] = None, location: Optional[str] = None, category: Optional[str] = None):
    jobs = list(ARTIFACTS.job_lookup.values())
    def _match(job: dict) -> bool:
        if query and query.lower() not in str(job.get(JOB_TITLE_COL, "")).lower() and query.lower() not in str(job.get(JOB_DESC_COL, "")).lower():
            return False
        if location and location.lower() not in str(job.get(JOB_LOCATION_COL, "")).lower():
            return False
        if category and category.lower() not in str(job.get("category", "")).lower():
            return False
        return True

    filtered = [job for job in jobs if _match(job)]
    start = (page - 1) * page_size
    end = start + page_size
    sliced = filtered[start:end]
    return JobListResponse(
        items=[
            JobOut(
                job_id=str(job.get(JOB_ID_COL)),
                job_title=str(job.get(JOB_TITLE_COL, "")),
                company=str(job.get("company", "")),
                location=str(job.get(JOB_LOCATION_COL, "")),
                category=str(job.get("category", "")),
                salary=str(job.get("salary", "")),
                descriptions=str(job.get(JOB_DESC_COL, "")),
            )
            for job in sliced
        ],
        page=page,
        page_size=page_size,
        total=len(filtered),
    )


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str):
    job = ARTIFACTS.job_lookup.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobOut(
        job_id=str(job.get(JOB_ID_COL)),
        job_title=str(job.get(JOB_TITLE_COL, "")),
        company=str(job.get("company", "")),
        location=str(job.get(JOB_LOCATION_COL, "")),
        category=str(job.get("category", "")),
        salary=str(job.get("salary", "")),
        descriptions=str(job.get(JOB_DESC_COL, "")),
    )


@app.post("/jobs", response_model=JobOut)
def post_job(payload: PostJobRequest, user=Depends(get_current_user)):
    if user.get("role") != "recruiter":
        raise HTTPException(status_code=403, detail="Only recruiters can post jobs")

    new_id = secrets.token_hex(6)
    job_record = {
        JOB_ID_COL: new_id,
        JOB_TITLE_COL: payload.job_title,
        "company": payload.company,
        JOB_LOCATION_COL: payload.location,
        "category": payload.category,
        "salary": payload.salary,
        JOB_DESC_COL: payload.descriptions,
    }
    ARTIFACTS.job_lookup[new_id] = job_record
    custom_jobs = load_json(CUSTOM_JOBS_FILE, [])
    custom_jobs.append(job_record)
    save_json(CUSTOM_JOBS_FILE, custom_jobs)

    # TODO: Re-train TF-IDF/LightFM to include newly posted jobs in recommendations.
    return JobOut(
        job_id=new_id,
        job_title=payload.job_title,
        company=payload.company,
        location=payload.location,
        category=payload.category,
        salary=payload.salary,
        descriptions=payload.descriptions,
    )


# ----------------------
# Applications & saved jobs (lightweight stubs)
# ----------------------
@app.post("/applications", status_code=201)
def apply_to_job(payload: ApplyRequest, user=Depends(get_current_user)):
    if payload.job_id not in ARTIFACTS.job_lookup:
        raise HTTPException(status_code=404, detail="Job not found")

    application = {
        "id": secrets.token_hex(6),
        "job_id": payload.job_id,
        "user_id": user["id"],
        "status": "submitted",
        "cover_letter": payload.cover_letter,
        "created_at": datetime.utcnow().isoformat(),
    }
    APPLICATIONS_DB.append(application)
    save_json(APPLICATIONS_FILE, APPLICATIONS_DB)
    return {"message": "Application submitted", "application": application}


@app.get("/applications")
def list_applications(user=Depends(get_current_user)):
    if user.get("role") == "recruiter":
        return [app for app in APPLICATIONS_DB if ARTIFACTS.job_lookup.get(app.get("job_id"))]
    return [app for app in APPLICATIONS_DB if app.get("user_id") == user["id"]]


@app.post("/saved/{job_id}")
def save_job(job_id: str, user=Depends(get_current_user)):
    if job_id not in ARTIFACTS.job_lookup:
        raise HTTPException(status_code=404, detail="Job not found")
    saved = SAVED_JOBS_DB.setdefault(user["id"], [])
    if job_id not in saved:
        saved.append(job_id)
    save_json(SAVED_JOBS_FILE, SAVED_JOBS_DB)
    return {"saved_jobs": saved}


@app.get("/saved")
def list_saved(user=Depends(get_current_user)):
    saved_ids = SAVED_JOBS_DB.get(user["id"], [])
    return [ARTIFACTS.job_lookup[jid] for jid in saved_ids if jid in ARTIFACTS.job_lookup]


# ----------------------
# Recommendations
# ----------------------
@app.get("/users/{user_id}/recommendations", response_model=List[RecommendationOut])
def user_recommendations(user_id: str, top_k: int = 10):
    user_record = USERS_DB.get(user_id)
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")

    text = user_profile_text(user_record)
    rec_df = ARTIFACTS.recommend(user_id=user_id, user_text=text, top_k=top_k)

    results: List[RecommendationOut] = []
    for _, row in rec_df.iterrows():
        job_id = row[JOB_ID_COL]
        job_meta = ARTIFACTS.job_lookup.get(job_id, {})
        results.append(
            RecommendationOut(
                job_id=job_id,
                job_title=job_meta.get(JOB_TITLE_COL, ""),
                company=job_meta.get("company", ""),
                location=job_meta.get(JOB_LOCATION_COL, ""),
                category=job_meta.get("category", ""),
                salary=job_meta.get("salary", ""),
                final_score=float(row.get("final_score", 0.0)),
                content_score=float(row.get("content_score", 0.0)),
                lfm_score=float(row.get("lfm_score", 0.0)),
            )
        )
    return results


@app.get("/health")
def health():
    return {"status": "ok", "jobs": len(ARTIFACTS.job_lookup)}

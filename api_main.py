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

from dotenv import load_dotenv
load_dotenv()
import os

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL is not None, "DATABASE_URL not found in environment"


import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from algorithms.core.data_loading import (
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
from algorithms.core.models import (
    build_lightfm_dataset,
    build_tfidf_representations,
    compute_hybrid_scores,
    predict_lightfm_scores_for_user,
)
from algorithms.employer_recommender import recommend_candidates_for_job
from crud import applications as crud_applications
from crud import jobs as crud_jobs
from crud import saved_jobs as crud_saved_jobs
from crud import users as crud_users
from db.session import SessionLocal, get_db
from models.job import Job
from models.user import User


SECRET_KEY = "dev-secret-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


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
        with SessionLocal() as db:
            db_jobs = crud_jobs.list_jobs(db)
        if db_jobs:
            extra_rows = [
                {
                    JOB_ID_COL: job.id,
                    JOB_TITLE_COL: job.title,
                    JOB_DESC_COL: job.description or "",
                    JOB_LOCATION_COL: job.location or "",
                    "category": job.category or "",
                    "company": job.company or "",
                    "salary": job.salary or "",
                }
                for job in db_jobs
            ]
            extra_df = pd.DataFrame(extra_rows)
            extra_df[JOB_ID_COL] = extra_df[JOB_ID_COL].astype(str)
            raw_jobs = pd.concat(
                [raw_jobs, extra_df[~extra_df[JOB_ID_COL].isin(raw_jobs[JOB_ID_COL])]],
                ignore_index=True,
            )
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

        from algorithms.core.models import train_lightfm  # local import to avoid cycle

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
ARTIFACTS.load_and_train()


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


class CandidateRecommendationOut(BaseModel):
    user_id: str
    score: float


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
def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = crud_users.get_user_with_profile(db, user_id)
    if user is None:
        raise credentials_exception
    return user


def serialize_user(user: User) -> UserProfile:
    profile = user.profile
    return UserProfile(
        id=user.id,
        email=user.email,
        full_name=profile.full_name if profile else "",
        role=user.role,
        preferred_location=profile.location if profile else None,
        headline=profile.headline if profile else None,
        skills=profile.skills_text if profile else None,
        experience_years=profile.experience_years if profile else None,
    )


def user_profile_text(user: User) -> str:
    profile = user.profile
    tokens = [
        profile.full_name if profile else "",
        user.role,
        profile.location if profile else "",
        profile.headline if profile else "",
        profile.skills_text if profile else "",
        str(profile.experience_years) if profile and profile.experience_years is not None else "",
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
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    if crud_users.get_user_by_email(db, payload.email):
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = secrets.token_hex(8)
    user_record = crud_users.create_user(
        db=db,
        user_id=user_id,
        email=payload.email,
        password_hash=hash_password(payload.password),
        role=payload.role,
        full_name=payload.full_name,
        location=payload.preferred_location,
        headline=payload.headline,
        skills_text=payload.skills,
        experience_years=payload.experience_years,
    )

    access_token = create_access_token({"sub": user_id})
    return TokenResponse(access_token=access_token, token_type="bearer", user=serialize_user(user_record))


@app.post("/auth/login", response_model=TokenResponse)
def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user = crud_users.get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.password_hash or ""):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=access_token, token_type="bearer", user=serialize_user(user))


# ----------------------
# Job endpoints
# ----------------------
@app.get("/jobs", response_model=JobListResponse)
def list_jobs(request: Request, page: int = 1, page_size: int = 20, query: Optional[str] = None, location: Optional[str] = None, category: Optional[str] = None):
    start_time = time.time()
    source = request.headers.get("X-Source", "unknown")
    
    # Use in-memory lookup built at startup and updated on job creation.
    all_jobs = list(ARTIFACTS.job_lookup.values())
    
    def _match(job: dict) -> bool:
        if query and query.lower() not in str(job.get(JOB_TITLE_COL, "")).lower() and query.lower() not in str(job.get(JOB_DESC_COL, "")).lower():
            return False
        if location and location.lower() not in str(job.get(JOB_LOCATION_COL, "")).lower():
            return False
        if category and category.lower() not in str(job.get("category", "")).lower():
            return False
        return True

    filtered = [job for job in all_jobs if _match(job)]
    start = (page - 1) * page_size
    end = start + page_size
    sliced = filtered[start:end]
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"[{datetime.now()}] [GET /jobs] [source={source}] user_id=anonymous, page={page}, page_size={page_size}, query={query}, location={location}, category={category}")
    print(f"  → total_from_artifacts={len(ARTIFACTS.job_lookup)}, post_filter_count={len(filtered)}, paginated_result_count={len(sliced)}, elapsed_ms={elapsed_ms}, status_code=200")
    
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
def get_job(job_id: str, db: Session = Depends(get_db)):
    # Try database first
    db_job = crud_jobs.get_job(db, job_id)
    if db_job:
        job_dict = {
            JOB_ID_COL: db_job.id,
            JOB_TITLE_COL: db_job.title,
            JOB_DESC_COL: db_job.description or "",
            JOB_LOCATION_COL: db_job.location or "",
            "category": db_job.category or "",
            "company": db_job.company or "",
            "salary": db_job.salary or "",
        }
        # Update in-memory lookup for consistency
        ARTIFACTS.job_lookup[db_job.id] = job_dict
        
        return JobOut(
            job_id=db_job.id,
            job_title=db_job.title,
            company=db_job.company or "",
            location=db_job.location or "",
            category=db_job.category or "",
            salary=db_job.salary or "",
            descriptions=db_job.description or "",
        )
    
    # Fallback to in-memory lookup (CSV jobs)
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
def post_job(payload: PostJobRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if user.role != "recruiter":
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
    crud_jobs.create_job(
        db=db,
        job_id=new_id,
        title=payload.job_title,
        description=payload.descriptions,
        location=payload.location,
        category=payload.category,
        company=payload.company,
        salary=payload.salary,
        employer_user_id=user.id,
        skills_text=None,
    )
    ARTIFACTS.job_lookup[new_id] = job_record

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


@app.get("/recruiter/jobs", response_model=List[JobOut])
def get_recruiter_jobs(request: Request, user=Depends(get_current_user), db: Session = Depends(get_db)):
    """Get all jobs posted by the current recruiter."""
    start_time = time.time()
    source = request.headers.get("X-Source", "unknown")
    
    if user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Only recruiters can access this endpoint")

    from sqlalchemy import select
    jobs = db.execute(select(Job).where(Job.employer_user_id == user.id)).scalars().all()
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    print(f"[{datetime.now()}] [GET /recruiter/jobs] [source={source}] user_id={user.id}")
    print(f"  → query_time_ms={elapsed_ms}, jobs_count={len(jobs)}, status_code=200")

    return [
        JobOut(
            job_id=str(job.id),
            job_title=job.title or "",
            company=job.company or "",
            location=job.location or "",
            category=job.category or "",
            salary=job.salary or "",
            descriptions=job.description or "",
        )
        for job in jobs
    ]



# ----------------------
# Applications & saved jobs (lightweight stubs)
# ----------------------
@app.post("/applications", status_code=201)
def apply_to_job(payload: ApplyRequest, user=Depends(get_current_user), db: Session = Depends(get_db)):
    job = crud_jobs.ensure_job_from_lookup(db, payload.job_id, ARTIFACTS.job_lookup)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    application = crud_applications.create_application(
        db=db,
        application_id=secrets.token_hex(6),
        user_id=user.id,
        job_id=payload.job_id,
        status="submitted",
        cover_letter=payload.cover_letter,
    )
    return {
        "message": "Application submitted",
        "application": {
            "id": application.id,
            "job_id": application.job_id,
            "user_id": application.user_id,
            "status": application.status,
            "cover_letter": application.cover_letter,
            "created_at": application.created_at.isoformat(),
        },
    }


@app.get("/applications")
def list_applications(user=Depends(get_current_user), db: Session = Depends(get_db)):
    if user.role == "recruiter":
        apps = crud_applications.list_applications_all(db)
        return [
            {
                "id": app.id,
                "job_id": app.job_id,
                "user_id": app.user_id,
                "status": app.status,
                "cover_letter": app.cover_letter,
                "created_at": app.created_at.isoformat(),
            }
            for app in apps
            if ARTIFACTS.job_lookup.get(app.job_id)
        ]
    apps = crud_applications.list_applications_by_user(db, user.id)
    return [
        {
            "id": app.id,
            "job_id": app.job_id,
            "user_id": app.user_id,
            "status": app.status,
            "cover_letter": app.cover_letter,
            "created_at": app.created_at.isoformat(),
        }
        for app in apps
    ]


@app.post("/saved/{job_id}")
def save_job(job_id: str, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if job_id not in ARTIFACTS.job_lookup:
        raise HTTPException(status_code=404, detail="Job not found")
    crud_jobs.ensure_job_from_lookup(db, job_id, ARTIFACTS.job_lookup)
    saved_ids = crud_saved_jobs.save_job(db, user.id, job_id)
    return {"saved_jobs": saved_ids}


@app.get("/saved")
def list_saved(user=Depends(get_current_user), db: Session = Depends(get_db)):
    saved_ids = crud_saved_jobs.list_saved_jobs(db, user.id)
    return [ARTIFACTS.job_lookup[jid] for jid in saved_ids if jid in ARTIFACTS.job_lookup]


# ----------------------
# Recommendations
# ----------------------
@app.get("/users/{user_id}/recommendations", response_model=List[RecommendationOut])
def user_recommendations(request: Request, user_id: str, top_k: int = 10, db: Session = Depends(get_db)):
    start_time = time.time()
    source = request.headers.get("X-Source", "unknown")
    
    user_record = crud_users.get_user_with_profile(db, user_id)
    if not user_record:
        raise HTTPException(status_code=404, detail="User not found")

    text = user_profile_text(user_record)
    user_exists_in_lf = user_id in getattr(ARTIFACTS, 'user_id_map', {})
    
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
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    scores = rec_df["final_score"].values if len(rec_df) > 0 else []
    print(f"[{datetime.now()}] [GET /recommendations] [source={source}] user_id={user_id}")
    print(f"  → user_exists_in_lf_dataset={user_exists_in_lf}, input_text_length={len(text)}")
    if len(scores) > 0:
        print(f"  → final_scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}, count={len(scores)}")
    print(f"  → returned_count={len(results)}, query_time_ms={elapsed_ms}, status_code=200")
    
    return results


@app.get("/employer/jobs/{job_id}/recommendations", response_model=List[CandidateRecommendationOut])
def employer_recommendations(job_id: str, top_n: int = 10, user=Depends(get_current_user), db: Session = Depends(get_db)):
    if user.role != "recruiter":
        raise HTTPException(status_code=403, detail="Only recruiters can request candidate recommendations")
    try:
        recs = recommend_candidates_for_job(job_id=job_id, db_session=db, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [CandidateRecommendationOut(user_id=rec["user_id"], score=rec["score"]) for rec in recs]


@app.get("/health")
def health():
    return {"status": "ok", "jobs": len(ARTIFACTS.job_lookup)}

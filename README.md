# Job Recommender

Hybrid job recommendation platform with a Flutter app (job seeker + recruiter flows), FastAPI backend, PostgreSQL database, and ML models (TF-IDF + LightFM hybrid, with SBERT/NCF experiments).

## What’s inside
- **Frontend (Flutter)**: Role selection, seeker dashboard, recruiter dashboard with jobs, applicants, company profile, and sessioned auth.
- **Backend (FastAPI)**: Auth, jobs CRUD, applications, saved jobs, `/users/{id}/recommendations` hybrid scoring endpoint.
- **Algorithms**: TF-IDF content similarity + LightFM collaborative filtering with weighted blend; optional SBERT/NCF experiments; evaluation scripts and plots.
- **Database**: PostgreSQL via SQLAlchemy/Alembic; entities for users, jobs, applications, saved jobs, user profiles.

## Run the Flutter app
```bash
flutter pub get
flutter run -d chrome   # or iOS/Android device
```

## Run the API locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # ensure fastapi, uvicorn, pandas, scikit-learn, lightfm, python-jose[cryptography]
uvicorn api_main:app --reload
```
API defaults to http://localhost:8000.

## Train/evaluate recommendations
```bash
# Train TF-IDF + LightFM hybrid and sample outputs
python algorithms/training/train_hybrid.py

# Full evaluation with metrics (Precision@K, Recall@K, NDCG@K, HitRate@K, MAP@K)
python evaluate_models.py

# Alpha tuning for content/collab blend
python evaluate_models.py --alpha-tuning
```
Artifacts and metrics are saved under `algorithms/data/` and figures under `algorithms/figures/`.

## Evaluation protocol (summary)
- Leave-one-out per user, negative sampling (N=99 configurable), candidate filtering by preferred location/target role.
- Metrics at K ∈ {1,5,10} plus legacy @10 columns.

## Key paths
- Frontend: `lib/` (providers, pages, widgets, theme)
- API: `api_main.py`, `crud/`, `models/`, `db/`
- Algorithms: `algorithms/` (core, models, training, evaluation, analysis)
- Data samples: `algorithms/data/`

## Notes
- Default hybrid weights: content 0.6 / LightFM 0.4 (configurable).
- Cold-start users get zero LightFM scores; new jobs may have zero TF-IDF until retrained.


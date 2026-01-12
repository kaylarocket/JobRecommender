# AI Copilot Instructions for Job Recommender System

## Architecture Overview

This is a **full-stack hybrid job recommendation platform** combining Flutter mobile/web frontend, FastAPI backend, PostgreSQL database, and ML algorithms.

### Major Components

1. **Frontend** (`lib/`): Flutter app with Provider state management
   - Auth provider handles user login/registration via `/auth/*` endpoints
   - Job provider manages job listings and recommendations
   - API client (`ApiService`) wraps HTTP calls with JWT token attachment

2. **Backend** (`api_main.py`): FastAPI server providing REST endpoints
   - Auth: register, login (JWT-based with HS256)
   - Jobs: CRUD operations + recommendations endpoint (`/users/{id}/recommendations`)
   - Hybrid scoring engine: TF-IDF + LightFM blended scores

3. **Algorithms** (`algorithms/`): ML pipeline for recommendations
   - **TF-IDF (content-based)**: Job text (title+description) vs user profile text (skills+degree)
   - **LightFM (collaborative)**: Matrix factorization on synthetic interactions; trained via WARP loss
   - **Hybrid scoring** (`compute_hybrid_scores`): Normalized min-max scaling + weighted blend with alpha parameter (default 0.6 for content)
   - Supports SBERT and NCF models for experimental comparison

4. **Database** (`models/`, `db/`): SQLAlchemy ORM with User, Job, Application, SavedJob entities

### Critical Data Flow

```
User Query → FastAPI → compute_recommendations() 
  ↓
TF-IDF similarity (precomputed job_tfidf matrix) + LightFM predict (user_id lookup)
  ↓
Min-max normalization → Weighted blend (alpha) → Top-K ranking
  ↓
JSON response with content_score, lfm_score, final_score, hybrid_score
```

**Key Constants** (in `algorithms/core/data_loading.py`):
- `MAX_USERS = 2000`, `MAX_JOBS = 5000` (dataset limits for local testing)
- `ALPHA_CONTENT = 0.6`, `ALPHA_LFM = 0.4` (hybrid weights—adjust in training code or API call)

## Essential Workflows

### Running Recommendations Training
```bash
# Train TF-IDF + LightFM and generate sample outputs
python algorithms/training/train_hybrid.py
# Output: algorithms/data/sample_user_recommendations.csv

# Full evaluation with metrics (Precision@K, NDCG@K, etc.)
python evaluate_models.py
```

### Starting the API Server
```bash
# Install: pip install fastapi uvicorn pandas scikit-learn lightfm python-jose[cryptography]
uvicorn api_main:app --reload  # Runs on http://localhost:8000
```

### Running Flutter App
```bash
flutter pub get
flutter run -d chrome  # or android/ios device
```

### Database Migrations
```bash
# Alembic setup (SQLAlchemy migrations)
alembic upgrade head  # Apply migrations
alembic revision --autogenerate -m "migration message"  # New migration
```

## Code Patterns & Conventions

### Hybrid Scoring Pattern
Every recommendation request follows this exact sequence (see `api_main.py:compute_recommendations`):

```python
# 1. Compute content scores (TF-IDF cosine similarity)
content_scores = compute_content_scores_for_user(user_idx, job_tfidf, user_tfidf)

# 2. Compute LF collab scores (return zeros if user unknown)
lfm_scores = predict_lightfm_scores_for_user(user_id, model, dataset, ...)

# 3. Blend with alpha (default 0.6); returns (hybrid, content_norm, lfm_norm)
hybrid, content_norm, lfm_norm = compute_hybrid_scores(content_scores, lfm_scores, alpha)

# 4. Rank and return top-K
df['final_score'] = hybrid
df.sort_values('final_score', ascending=False).head(top_k)
```

**Important**: When modifying scoring, edit `compute_hybrid_scores()` in `algorithms/models/hybrid_model.py`, not inline API logic.

### Data Column Mapping
These constants **must stay in sync** across data loading, training, and API inference:

```python
# From algorithms/core/data_loading.py
JOB_ID_COL, JOB_TITLE_COL, JOB_DESC_COL, JOB_LOCATION_COL, JOB_CATEGORY_COL
USER_ID_COL, USER_SKILLS_COL, USER_DEGREE_COL, USER_PREFERRED_LOC_COL, USER_TARGET_ROLE_COL
```

Changing CSV column names requires updating all three: data loading, model training, and API artifact lookup.

### API Authentication
- Uses FastAPI `OAuth2PasswordBearer` with JWT tokens (HS256, 24h expiry)
- Passwords hashed with SHA-256 + salt (see `hash_password`/`verify_password`)
- Every protected endpoint requires `Authorization: Bearer <token>` header (ApiService handles this)

### Database Relationships
- **User** (job_seeker or recruiter) → **SavedJob** (many), **Application** (many), **UserProfile** (one)
- **User** (recruiter) → **Job** (many postings)
- **Job** → **SavedJob**, **Application** (both with cascade delete)

## Project-Specific Conventions

1. **Pandas DataFrames as Intermediate Format**: All algorithms work with pandas—see `build_job_table()` and `build_user_table()` for feature engineering patterns.

2. **Feature Text Concatenation**: User and job features are simply concatenated as strings for TF-IDF. No one-hot encoding or categorical handling—extend via `build_user_table()` if needed.

3. **Proxy User Mapping**: Unknown users default to zero LightFM scores (cold-start fallback). User IDs are mapped via `user_id_map` dict in `HybridArtifacts.load_and_train()`.

4. **Synthetic Interactions**: Generated via `build_synthetic_interactions()` from text similarity—not from real user clicks. See ALGORITHM_ANALYSIS.md for limitations.

5. **CSV-Driven Evaluation**: Training scripts read `jobstreet_all_jobs.csv` and `job_applicants.csv` from `algorithms/data/`. External jobs (from API database) are merged at recommendation time.

## Testing & Evaluation

### Evaluation Protocol
- **Leave-One-Out CV**: For each user, one interaction held back as test, rest used for training
- **Negative Sampling**: 99 candidate jobs per test interaction (N=99 configurable via `--negative-sample-size`)
- **Filtering**: Candidates filtered by preferred location and target role to match real use case
- **Metrics Computed**: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MAP@K at K={1,5,10}

### Evaluation Metrics Explained
- **Precision@K** = (hits in top-K) / K → measures precision of top-K recommendations
- **Recall@K** = (hits in top-K) / (total positive items) → measures coverage of held-out items
- **NDCG@K** (Normalized DCG) = DCG / ideal DCG → ranks item relevance; 1.0 = perfect ranking
- **Hit Rate@K** = 1 if any hit in top-K else 0 → binary success metric
- **MAP@K** = average precision across all relevant positions → rewards correct items early

### Alpha Tuning
The hybrid model blends TF-IDF (content) and LightFM (collaborative) scores. To find optimal balance:
```bash
python evaluate_models.py --alpha-tuning  # Tests alpha in [0.0, 0.3, 0.5, 0.7, 1.0]
# Output: algorithms/data/alpha_tuning_results.csv
# Default ALPHA_CONTENT = 0.6 (60% content, 40% collaborative)
```

### Running Evaluations
```bash
# Full evaluation with all models
python evaluate_models.py

# With custom negative sample size
python evaluate_models.py --negative-sample-size 49

# Output files
algorithms/data/evaluation_results.csv        # All models' metrics at K={1,5,10}
algorithms/data/alpha_tuning_results.csv      # Alpha sensitivity analysis
algorithms/data/threshold_results.csv         # Score threshold analysis
```

### Visualization
```bash
python algorithms/analysis/plot_results.py  # Generates PNG figures in algorithms/figures/
```

### Inspecting Model Outputs
```python
# Debug TF-IDF content scores for a user
from algorithms.core.data_loading import build_job_table, build_user_table
from algorithms.models.tfidf_model import build_tfidf_representations, compute_content_scores_for_user

jobs = build_job_table(raw_jobs)
users = build_user_table(raw_users)
_, job_tfidf, user_tfidf = build_tfidf_representations(jobs, users)
content_scores = compute_content_scores_for_user(user_idx=0, job_tfidf=job_tfidf, user_tfidf=user_tfidf)
print(f"Content scores shape: {content_scores.shape}, top 10: {np.argsort(content_scores)[-10:]}")

# Compare hybrid vs individual scores at recommendation time
# In api_main.py compute_recommendations():
# rec_df['content_score'] + rec_df['lfm_score'] + rec_df['final_score'] all available before ranking
```

### Debugging API Mismatches
If API recommendations differ from training outputs, check:
1. **User ID mapping mismatch**: Unknown user_ids default to zero LightFM scores (cold-start)
   - Look for `if user_id in user_id_map` in `HybridArtifacts.compute_recommendations()`
2. **Job merging**: External jobs from database merged at runtime via `extra_df`
   - New jobs added after training won't have TF-IDF vectors; they get zeros
3. **Alpha mismatch**: Training uses ALPHA_CONTENT=0.6 by default; API call may override via parameter
4. **Feature mismatch**: Ensure CSV columns match constants in `data_loading.py`

## Debugging & Profiling

### Debugging Recommendation Mismatches

If an API recommendation doesn't match training output or seems off:

**1. Verify user is known**
```python
from api_main import ARTIFACTS

user_id = "my_user_id"
user_idx = ARTIFACTS.user_id_map.get(user_id)
if user_idx is None:
    print("Cold-start user: LightFM scores will be zeros")
else:
    print(f"User index: {user_idx}")
```

**2. Inspect individual scores**
```python
from algorithms.models.tfidf_model import compute_content_scores_for_user
from algorithms.models.lightfm_model import predict_lightfm_scores_for_user

content = compute_content_scores_for_user(user_idx, ARTIFACTS.job_tfidf, ARTIFACTS.user_tfidf)
lfm = predict_lightfm_scores_for_user(user_id, ARTIFACTS.model, ARTIFACTS.dataset, ...)

print(f"Content top-5: {np.argsort(content)[-5:]}")
print(f"LightFM top-5: {np.argsort(lfm)[-5:]}")
print(f"Content scores: min={content.min():.3f}, max={content.max():.3f}, mean={content.mean():.3f}")
print(f"LightFM scores: min={lfm.min():.3f}, max={lfm.max():.3f}, mean={lfm.mean():.3f}")
```

**3. Check for new jobs (not in training set)**
```python
# New jobs added to database after training have zero TF-IDF vectors
rec_df = ARTIFACTS.compute_recommendations(user_id, top_k=20)
print(rec_df[['job_id', 'content_score', 'lfm_score', 'final_score']].head())
# If content_score == 0 and lfm_score == 0, job is new
```

**4. Verify alpha blending**
```python
from algorithms.models.hybrid_model import compute_hybrid_scores

alpha = 0.6
hybrid, content_norm, lfm_norm = compute_hybrid_scores(content, lfm, alpha)
print(f"Hybrid formula: {alpha} * {content_norm[0]:.3f} + {1-alpha} * {lfm_norm[0]:.3f} = {hybrid[0]:.3f}")
```

### Profiling Recommendation Latency

```python
import time

start = time.time()
recs = ARTIFACTS.compute_recommendations("user_123", top_k=10)
elapsed = time.time() - start
print(f"Recommendation time: {elapsed:.3f}s")

# If too slow, check:
# 1. TF-IDF vectorization (for new jobs): may regenerate if dataset changed
# 2. LightFM predict call: should be O(n_jobs) matrix multiply
# 3. Sorting: O(n_jobs * log(k)) for top-k, usually negligible
```

### Checking Model Artifact Sizes

```python
import sys

print(f"Job TF-IDF matrix: {ARTIFACTS.job_tfidf.nbytes / 1e6:.1f} MB")
print(f"User TF-IDF matrix: {ARTIFACTS.user_tfidf.nbytes / 1e6:.1f} MB")
print(f"LightFM model params: {sum(p.numel() for p in ARTIFACTS.model.parameters()) if hasattr(ARTIFACTS.model, 'parameters') else 'N/A'}")
print(f"Vectorizer vocabulary: {len(ARTIFACTS.vectorizer.get_feature_names_out()) if ARTIFACTS.vectorizer else 'N/A'}")
```

### Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| All scores are zeros | User is new (cold-start) | Expected; LightFM defaults to zero, TF-IDF also zero if no profile text |
| Recommendations unchanged after training | Old artifacts not reloaded | Restart API: `uvicorn api_main:app --reload` |
| Different alpha, same results | Alpha not passed to endpoint | Check `/users/{id}/recommendations?alpha=0.7` query param is used |
| Slow recommendations | Large job dataset | Reduce MAX_JOBS in data_loading.py or profile TF-IDF computation |
| 401 Unauthorized from Flutter | Token expired or not attached | Check ApiService._headers() and token lifecycle in AuthProvider |
| Database migrations fail | Alembic head mismatch | Run `alembic upgrade head` or `alembic downgrade -1` then up |

### AuthProvider Pattern
**File**: `lib/providers/auth_provider.dart`

```dart
class AuthProvider extends ChangeNotifier {
  final ApiService _apiService;
  UserSession? _session;
  bool _loading = false;
  String? _error;

  Future<void> login(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _session = await _apiService.login(email: email, password: password);
      // Token automatically attached to ApiService after login
      _apiService.updateToken(_session?.token);
    } catch (e) {
      _error = e.toString();
    } finally {
      _loading = false;
      notifyListeners();
    }
  }

  // TODO: Persist session using SharedPreferences for offline support
}
```

**Key Pattern**: UI observes `isLoading`, `error`, `isAuthenticated` via Consumer widgets  
**Error Handling**: Catch and display errors in UI while keeping state consistent  
**Token Management**: `_apiService.updateToken()` called after successful login; used in all subsequent requests

### JobProvider Pattern
**File**: `lib/providers/job_provider.dart`

```dart
class JobProvider extends ChangeNotifier {
  Future<void> refreshRecommendations(String userId) async {
    try {
      recommendations = await _apiService.getRecommendations(userId);
      notifyListeners();
    } catch (e) {
      // Fail gracefully: keep stale recommendations in UI, show error banner
      error = e.toString();
      notifyListeners();
    }
  }
}
```

**Resilience Pattern**: Recommendations failure doesn't block job browsing; recommendations appear empty if API unreachable

### API Call Flow
```dart
// 1. ApiService._headers() attaches token if available
Map<String, String> _headers({bool auth = false}) {
  return {
    'Content-Type': 'application/json',
    if (auth && _token != null) 'Authorization': 'Bearer $_token',
  };
}

// 2. Every authenticated request uses: _headers(auth: true)
final resp = await http.get(
  Uri.parse('$baseUrl/users/$userId/recommendations'),
  headers: _headers(auth: true),
);

// 3. If 401 Unauthorized, consider refresh token or redirect to login
```

**Token Lifecycle**:
- On `login`: JWT stored, passed to `updateToken()`
- On all requests: Attached as `Authorization: Bearer <token>`
- On expiry (24h): API returns 401 → Provider catches error, suggests re-login
- On `logout`: Token cleared via `updateToken(null)`

### Cross-Component Communication

| Component | Communicates Via | Notes |
|-----------|-----------------|-------|
| Flutter → FastAPI | HTTP + JWT token | ApiService wraps all calls; token auto-attached |
| AuthProvider ↔ JobProvider | ChangeNotifier | Both receive token updates from ApiService |
| API ↔ Algorithms | Python imports | `HybridArtifacts` singleton, trained at startup |
| API ↔ Database | SQLAlchemy ORM | Session from `db.session.SessionLocal()` |
| Training Scripts | Direct pandas/numpy | No API—standalone Python execution |

## Key Files to Reference

- **Core recommendation logic**: `algorithms/core/models.py` (imports forwarding to `algorithms/models/*`)
- **Hybrid scorer**: `algorithms/models/hybrid_model.py`
- **API endpoints**: `api_main.py` lines ~250–500 (auth, jobs, recommendations)
- **Flutter state**: `lib/providers/auth_provider.dart`, `lib/providers/job_provider.dart`
- **Database models**: `models/*.py` (User, Job, Application, SavedJob, UserProfile, Employer)

## Future Extension Points

### Experimental Models (SBERT & NCF)

The codebase includes two alternative models for comparison; they are **not** used in production recommendations (TF-IDF + LightFM is primary).

#### SBERT (Sentence-BERT)
**File**: `algorithms/models/sbert_model.py`  
**When to use**: Job descriptions with semantic meaning; better for contextual similarity than keyword matching.

```bash
pip install sentence-transformers torch
```

```python
from algorithms.models.sbert_model import build_sbert_representations, compute_sbert_scores_for_user

# Requires: transformers + torch (heavy dependencies)
model, job_embeddings, user_embeddings = build_sbert_representations(
    users=user_texts,
    jobs=job_texts,
    model_name="all-MiniLM-L6-v2",  # Pre-trained SBERT model
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Cosine similarity on embeddings
scores = compute_sbert_scores_for_user(user_idx, job_embeddings, user_embeddings)
```

**Pros**: Semantic understanding, multilingual, transfer learning  
**Cons**: Slower inference, high memory, requires GPU for production  
**Use Case**: Fine-tuning on domain-specific job descriptions; semantic search

#### NCF (Neural Collaborative Filtering)
**File**: `algorithms/models/ncf_model.py`  
**When to use**: Large user-job interaction dataset; learning non-linear preference patterns.

```bash
pip install torch
```

```python
from algorithms.models.ncf_model import build_ncf_training_data, train_ncf_model, predict_ncf_scores_for_user

# Build training data from interactions
user_item_pairs, labels = build_ncf_training_data(interactions_df, jobs)

# Train MLP-based model
model = train_ncf_model(
    user_item_pairs=user_item_pairs,
    labels=labels,
    n_users=len(users),
    n_items=len(jobs),
    embedding_dim=32,
    hidden_layers=(64, 32),
    epochs=10,
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Predict for a user
scores = predict_ncf_scores_for_user(user_id, model, jobs)
```

**Pros**: Learns non-linear interactions, embeddings capture user preferences  
**Cons**: Requires large interaction dataset, overfits on small data, cold-start issues  
**Use Case**: Production with mature user base; A/B testing against hybrid baseline

#### Evaluation Comparison
Run all models side-by-side:
```bash
python evaluate_models.py  # Compares TF-IDF, SBERT, LightFM, NCF, Hybrid
# Output: algorithms/data/evaluation_results.csv with all metrics
```

**Output columns**: model_name, metric_name (Precision@1, Recall@5, NDCG@10, etc.), score

### Adding a New Recommender

1. **Create model file** in `algorithms/models/my_model.py`
   - Implement: `build_representations()`, `compute_scores_for_user()`, `train_my_model()`
   - Follow SBERT/NCF pattern for imports and error handling

2. **Register in evaluation** at `algorithms/training/evaluate_models.py`
   - Add to `main()` function's model comparison loop
   - Define train/predict functions and evaluation parameters

3. **Integrate into API** (optional for production)
   - Add to `HybridArtifacts.load_and_train()` in `api_main.py`
   - Create `/users/{id}/recommendations?model=my_model` endpoint
   - Ensure feature columns match `data_loading.py` constants

4. **Test evaluation metrics**
   - Run `python evaluate_models.py` and verify CSV outputs match expectations

### Tuning Alpha (Hybrid Weights)

The default `ALPHA_CONTENT = 0.6` weights TF-IDF (60%) vs LightFM (40%). To find optimal balance:

**1. Offline tuning** (training phase):
```bash
python evaluate_models.py --alpha-tuning
# Tests alpha in [0.0, 0.3, 0.5, 0.7, 1.0]
# Output: algorithms/data/alpha_tuning_results.csv
```

**2. Update training script** (if optimal alpha differs):
```python
# In algorithms/training/train_hybrid.py
ALPHA_CONTENT = 0.7  # If 0.7 performs best
```

**3. Update API** (to use new alpha):
```python
# In api_main.py, line ~250
alpha = request.alpha if request.alpha is not None else 0.7  # Updated default
```

**4. Online A/B testing** (production):
- Pass `alpha` parameter to `/users/{id}/recommendations?alpha=0.7`
- Compare CTR/conversion between alpha groups

### Real User Interactions

Currently, `build_synthetic_interactions()` generates interactions from text similarity (not real clicks). To switch:

1. **Replace synthetic generation** in `api_main.py`:
```python
# OLD: interactions_df = build_synthetic_interactions(self.users_features, self.jobs_features)

# NEW: Load from database
interactions_df = []
for user in users:
    for application in user.applications:
        interactions_df.append({'user_id': user.id, 'job_id': application.job_id, 'weight': 1.0})
interactions_df = pd.DataFrame(interactions_df)
```

2. **Retrain LightFM** with real interactions:
```bash
python algorithms/training/train_hybrid.py  # Automatically uses new interaction source
```

3. **No model code changes needed**—LightFM API remains identical

### Feature Engineering

Extend `build_user_table()` and `build_job_table()` in `algorithms/core/data_loading.py`:

```python
def build_user_table(users: pd.DataFrame) -> pd.DataFrame:
    # Current: concatenate skills + degree
    users['text'] = users[USER_SKILLS_COL].fillna('') + ' ' + users[USER_DEGREE_COL].fillna('')
    
    # NEW: Add seniority bucketing
    users['seniority'] = pd.cut(users[USER_YEARS_PRO_COL], bins=[0, 2, 5, 10, 100], 
                                 labels=['Junior', 'Mid', 'Senior', 'Lead'])
    users['text'] = users['text'] + ' ' + users['seniority'].astype(str)
    
    # NEW: Add location context
    users['text'] = users['text'] + ' ' + users[USER_PREFERRED_LOC_COL].fillna('')
    
    return users
```

**Note**: TF-IDF will treat new text features as additional vocabulary; no retraining required—just rebuild CSVs and retrain models.

1. **New Models**: Add to `algorithms/models/`, import in `algorithms/core/models.py`
2. **New API Features**: Add endpoints to `api_main.py`, corresponding CRUD in `crud/`
3. **Real Interactions**: Replace `build_synthetic_interactions()` with actual user feedback; LightFM code remains unchanged

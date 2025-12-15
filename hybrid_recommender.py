"""
Hybrid Job Recommender System
- Content-based filtering: TF-IDF + Cosine Similarity
- Collaborative filtering: LightFM (Matrix Factorization)
- Data: JobStreet Job Postings + 70k Job Applicants Dataset

Main pipeline:
1) Load & clean data
2) Feature engineering:
   - Jobs: title + description + category
   - Users: skills + degree + main branch + experience + skill level
3) Content-based similarity (TF-IDF + cosine)
4) Collaborative filtering (LightFM WARP) on synthetic implicit feedback
5) Score normalization (min-max) and hybrid combination
6) Export sample recommendations to CSV

Output: sample_user_recommendations.csv
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from lightfm import LightFM
from lightfm.data import Dataset


# ================================
# CONFIGURATION (adjust here only)
# ================================

USE_KAGGLEHUB = False  # we use LOCAL CSV files

LOCAL_JOBSTREET_CSV = "jobstreet_all_jobs.csv"      # <--- ensure filename matches yours
LOCAL_APPLICANTS_CSV = "job_applicants.csv"         # <--- ensure filename matches yours

# JobStreet columns
JOB_ID_COL = "job_id"
JOB_TITLE_COL = "job_title"
JOB_DESC_COL = "descriptions"
JOB_LOCATION_COL = "location"
JOB_CATEGORY_COL = "category"

# Applicants columns
USER_ID_COL = "Unnamed: 0"
USER_SKILLS_COL = "HaveWorkedWith"
USER_DEGREE_COL = "EdLevel"
USER_PREFERRED_LOC_COL = "Country"
USER_TARGET_ROLE_COL = "MainBranch"
USER_YEARS_PRO_COL = "YearsCodePro"     # extra feature
USER_SKILL_LEVEL_COL = "ComputerSkills" # numeric-ish extra feature

# Limits for fast testing (reduce if too slow)
MAX_USERS = 2000
MAX_JOBS = 5000

# Hybrid weights
ALPHA_CONTENT = 0.6  # weight for TF-IDF cosine
ALPHA_LFM = 0.4      # weight for LightFM scores


# ================================
# 1. LOAD DATASETS
# ================================

def load_jobstreet_job_dataset():
    print(f"Loading JobStreet jobs from local CSV: {LOCAL_JOBSTREET_CSV}")
    df = pd.read_csv(LOCAL_JOBSTREET_CSV)
    print("JobStreet jobs – first 5 rows:")
    print(df.head())
    return df


def load_applicants_dataset():
    print(f"Loading Applicants from local CSV: {LOCAL_APPLICANTS_CSV}")
    df = pd.read_csv(LOCAL_APPLICANTS_CSV)
    print("Applicants – first 5 rows:")
    print(df.head())
    return df


# ================================
# 2. JOB TABLE + FEATURE ENGINEERING
# ================================

def build_job_table(raw_jobs: pd.DataFrame) -> pd.DataFrame:
    df = raw_jobs.copy()

    # Check required columns
    for col in [JOB_ID_COL, JOB_TITLE_COL, JOB_DESC_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' — Check CONFIG section.")

    df["job_id"] = df[JOB_ID_COL].astype(str)
    df["job_location"] = df.get(JOB_LOCATION_COL, "").fillna("").astype(str)
    df["job_category"] = df.get(JOB_CATEGORY_COL, "").fillna("").astype(str)

    # Feature engineering: combine multiple job attributes into one text
    df["job_text"] = (
        df[JOB_TITLE_COL].fillna("").astype(str)
        + " "
        + df[JOB_DESC_COL].fillna("").astype(str)
        + " "
        + df["job_category"].fillna("").astype(str)
    ).str.lower()

    return df[["job_id", "job_text", "job_location", "job_category"]]


# ================================
# 3. USER TABLE + FEATURE ENGINEERING
# ================================

def build_user_table(raw_users: pd.DataFrame) -> pd.DataFrame:
    df = raw_users.copy()

    # Base ID
    df["user_id"] = df[USER_ID_COL].astype(str)

    # Skills, degree, target role, preferred country
    df["skills_text"] = df.get(USER_SKILLS_COL, "").fillna("").astype(str)
    df["degree_text"] = df.get(USER_DEGREE_COL, "").fillna("").astype(str)
    df["target_role"] = df.get(USER_TARGET_ROLE_COL, "").fillna("").astype(str)
    df["preferred_location"] = df.get(USER_PREFERRED_LOC_COL, "").fillna("").astype(str)

    # Extra engineered features:
    #  - years of professional coding experience
    #  - high-level skill level
    df["years_token"] = "years_" + df.get(USER_YEARS_PRO_COL, "").fillna("").astype(str)
    df["skill_level_token"] = "skilllvl_" + df.get(USER_SKILL_LEVEL_COL, "").fillna("").astype(str)

    # Final user_text feature used for TF-IDF:
    #   skills + degree + main branch + experience + skill level
    df["user_text"] = (
        df["skills_text"]
        + " "
        + df["degree_text"]
        + " "
        + df["target_role"]
        + " "
        + df["years_token"]
        + " "
        + df["skill_level_token"]
    ).str.lower()

    return df[["user_id", "user_text", "preferred_location", "target_role"]].drop_duplicates("user_id")


# ================================
# 4. SYNTHETIC INTERACTIONS (IMPLICIT FEEDBACK)
# ================================

def build_synthetic_interactions(users: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Build synthetic implicit feedback because we don't have real logs.

    For each user:
      - tokenize user_text (skills + degree + role + exp + skill level)
      - find jobs whose job_text matches some of these tokens
      - bias towards jobs in the same Country
      - sample up to 10 positive interactions per user

    This creates a dense interaction matrix so LightFM can learn meaningful patterns.
    """
    rng = np.random.default_rng(seed=42)
    interactions = []

    jobs_local = jobs.copy()
    jobs_local["job_text_lower"] = jobs_local["job_text"].str.lower()
    jobs_local["job_location_lower"] = jobs_local["job_location"].str.lower()

    for _, user in users.iterrows():
        u = user["user_id"]
        pref_loc = str(user["preferred_location"]).lower()
        user_text = str(user["user_text"]).lower()

        # Basic tokenization: split by space, keep reasonably long tokens
        tokens = [t.strip() for t in user_text.split() if len(t) > 2]

        cand = jobs_local

        # Match some tokens in job_text
        if tokens:
            mask = False
            for t in tokens[:10]:  # limit to 10 tokens per user for speed
                mask = mask | cand["job_text_lower"].str.contains(t, regex=False)
            cand = cand[mask]

        # If no matches, fallback to all jobs
        if cand.empty:
            cand = jobs_local

        # Bias to same country/location
        if pref_loc:
            in_loc = cand[cand["job_location_lower"].str.contains(pref_loc)]
            if not in_loc.empty and rng.random() < 0.7:
                cand = in_loc

        # Ensure several interactions per user
        n_pos = min(10, len(cand))
        sampled = cand.sample(n=n_pos, random_state=None)

        for _, j in sampled.iterrows():
            interactions.append({"user_id": u, "job_id": j["job_id"], "weight": 1.0})

    interactions_df = pd.DataFrame(interactions)
    print(f"Synthetic interactions: {len(interactions_df)} rows")
    print("Example interactions per user:")
    print(interactions_df["user_id"].value_counts().head())
    return interactions_df


# ================================
# 5. TF-IDF + COSINE (CONTENT-BASED)
# ================================

def build_tfidf_representations(users: pd.DataFrame, jobs: pd.DataFrame):
    """
    TF-IDF feature learning:
      - Vectorize job_text + user_text in a shared space.
      - This implicitly normalizes term frequencies and downweights common words.
    """
    corpus = jobs["job_text"].tolist() + users["user_text"].tolist()

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    n_jobs = len(jobs)
    job_tfidf = tfidf_matrix[:n_jobs]
    user_tfidf = tfidf_matrix[n_jobs:]

    return vectorizer, job_tfidf, user_tfidf


def compute_content_scores_for_user(user_idx: int, job_tfidf, user_tfidf):
    """
    Compute cosine similarity between a user's TF-IDF vector and all job vectors.
    """
    return cosine_similarity(user_tfidf[user_idx], job_tfidf).ravel()


# ================================
# 6. LIGHTFM (COLLABORATIVE FILTERING)
# ================================

def build_lightfm_dataset(interactions_df, users, jobs):
    """
    Convert (user_id, job_id, weight) into a LightFM Dataset + interaction matrix.
    """
    dataset = Dataset()
    dataset.fit(
        users=users["user_id"].tolist(),
        items=jobs["job_id"].tolist(),
    )

    tuples = [
        (row["user_id"], row["job_id"], float(row["weight"]))
        for _, row in interactions_df.iterrows()
    ]

    interactions, weights = dataset.build_interactions(tuples)

    return dataset, interactions, weights


def train_lightfm(interactions, weights):
    """
    Train LightFM using WARP loss (good for implicit feedback & ranking).
    """
    model = LightFM(
        loss="warp",
        no_components=50,
        random_state=42
    )
    model.fit(interactions, sample_weight=weights, epochs=15, num_threads=4)
    return model


def predict_lightfm_scores_for_user(user_id, model, dataset, jobs):
    """
    Predict LightFM scores for a given user across all jobs.
    This correctly uses the dataset's internal ID mappings.
    """
    n_jobs = len(jobs)

    # Correct unpacking of mapping()
    user_id_map, _, item_id_map, _ = dataset.mapping()

    # Cold-start user: not in training interactions
    if user_id not in user_id_map:
        return np.zeros(n_jobs)

    # Map external user_id -> internal LightFM user index
    uid = user_id_map[user_id]

    # Map each job_id in our jobs DataFrame -> internal LightFM item index
    job_ids = jobs["job_id"].tolist()
    item_internal_ids = np.array([item_id_map[jid] for jid in job_ids])

    # Predict scores for this user across all jobs (in our DataFrame order)
    scores = model.predict(uid, item_internal_ids)
    return scores


# ================================
# 7. HYBRID SCORE + NORMALIZATION + RANKING
# ================================

def recommend_for_user(
    user_id,
    user_idx,
    users,
    jobs,
    job_tfidf,
    user_tfidf,
    model,
    dataset,
    topN=10,
    alpha=ALPHA_CONTENT
):
    """
    Hybrid recommendation:
      1) Content-based score = cosine(user_tfidf, job_tfidf)
      2) Collaborative score = LightFM(user, job)
      3) Normalize both using min-max scaling to [0, 1]
      4) Combine with linear weight:
           final = alpha * content_norm + (1 - alpha) * lfm_norm
    """
    n_jobs = len(jobs)

    # --- Content part ---
    content = compute_content_scores_for_user(user_idx, job_tfidf, user_tfidf)
    content_norm = minmax_scale(content) if n_jobs > 1 else content

    # --- Collaborative part ---
    lfm = predict_lightfm_scores_for_user(user_id, model, dataset, jobs)
    lfm_norm = minmax_scale(lfm) if n_jobs > 1 else lfm

    # --- Hybrid score ---
    final = alpha * content_norm + (1 - alpha) * lfm_norm

    df = jobs.copy()
    df["content_score"] = content_norm
    df["lfm_score"] = lfm_norm
    df["final_score"] = final

    return df.sort_values("final_score", ascending=False).head(topN)


# ================================
# 8. MAIN PIPELINE
# ================================

def main():
    # 1) Load raw data
    raw_jobs = load_jobstreet_job_dataset()
    raw_applicants = load_applicants_dataset()

    # 2) Reduce size for experiments
    if len(raw_jobs) > MAX_JOBS:
        raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
    if len(raw_applicants) > MAX_USERS:
        raw_applicants = raw_applicants.sample(MAX_USERS, random_state=42)

    # 3) Feature engineering → clean job & user tables
    jobs = build_job_table(raw_jobs).reset_index(drop=True)
    users = build_user_table(raw_applicants).reset_index(drop=True)

    print(f"Prepared {len(jobs)} jobs and {len(users)} users.")

    # 4) Synthetic interactions → implicit feedback
    interactions_df = build_synthetic_interactions(users, jobs)

    # 5) TF-IDF representations
    vectorizer, job_tfidf, user_tfidf = build_tfidf_representations(users, jobs)

    # 6) LightFM dataset + training
    dataset, interactions, weights = build_lightfm_dataset(interactions_df, users, jobs)
    model = train_lightfm(interactions, weights)

    # 7) Recommendations for 5 users (demo)
    all_recs = []
    for i in range(min(5, len(users))):
        user_id = users.loc[i, "user_id"]
        rec = recommend_for_user(
            user_id=user_id,
            user_idx=i,
            users=users,
            jobs=jobs,
            job_tfidf=job_tfidf,
            user_tfidf=user_tfidf,
            model=model,
            dataset=dataset,
            topN=10,
            alpha=ALPHA_CONTENT,
        )
        print("=" * 60)
        print(f"Top recommendations for user {user_id}:")
        print(rec[["job_id", "final_score", "content_score", "lfm_score"]])

        rec_out = pd.DataFrame({
            "user_id": user_id,
            "job_id": rec["job_id"],
            "rank": np.arange(1, len(rec) + 1),
            "final_score": rec["final_score"],
            "content_score": rec["content_score"],
            "lfm_score": rec["lfm_score"],
        })
        all_recs.append(rec_out)

    # 8 Save sample recommendations
    if all_recs:
        df_out = pd.concat(all_recs, ignore_index=True)
        df_out.to_csv("sample_user_recommendations.csv", index=False)
        print("Saved sample_user_recommendations.csv")


if __name__ == "__main__":
    main()

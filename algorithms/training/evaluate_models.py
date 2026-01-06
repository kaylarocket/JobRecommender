"""
Evaluation pipeline for comparing TF-IDF, SBERT, LightFM, NCF, and Hybrid recommenders.

README:
- Run evaluations: python3 evaluate_models.py
- Plot thesis figures: python3 algorithms/analysis/plot_results.py
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from algorithms.core.data_loading import (
    MAX_JOBS,
    MAX_USERS,
    build_job_table,
    build_synthetic_interactions,
    build_user_table,
    load_applicants_dataset,
    load_jobstreet_job_dataset,
)
from algorithms.core.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mean,
    minmax_normalize,
    ndcg_at_k,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
)
from algorithms.core.registry import register_model
from algorithms.models.hybrid_model import compute_hybrid_scores
from algorithms.models.lightfm_model import (
    build_lightfm_dataset,
    predict_lightfm_scores_for_user,
    train_lightfm,
)
from algorithms.models.ncf_model import (
    build_ncf_training_data,
    predict_ncf_scores_for_user,
    train_ncf_model,
)
from algorithms.models.sbert_model import (
    build_sbert_representations,
    compute_sbert_scores_for_user,
)
from algorithms.models.tfidf_model import (
    build_tfidf_representations,
    compute_content_scores_for_user,
)

TEST_SIZE = 0.2
TOP_K = 10
DEFAULT_ALPHA = 0.6
ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]
EVAL_SEED = 42
LEAVE_ONE_OUT = True
EVAL_KS = [1, 5, 10]

THRESHOLDS = [i / 100 for i in range(1, 51)]  # 0.01 ... 0.50
NEGATIVE_SAMPLE_SIZE = 99
ENABLE_EVAL_FILTERING = True

OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_test_split_interactions(
    interactions: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = 42,
    leave_one_out: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split per user for implicit interactions.
    Users with <2 interactions are kept entirely in train.
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    test_rows = []

    for _, group in interactions.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        if leave_one_out:
            test_sample = group.sample(n=1, random_state=rng.integers(0, 1_000_000))
        else:
            n_test = max(1, int(len(group) * test_size))
            test_sample = group.sample(n=n_test, random_state=rng.integers(0, 1_000_000))
        train_sample = group.drop(test_sample.index)
        train_rows.append(train_sample)
        test_rows.append(test_sample)

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=interactions.columns)
    return train_df, test_df


def _rank_jobs(
    scores: np.ndarray,
    jobs: pd.DataFrame,
    job_index: Dict[str, int],
    exclude_ids: Set[str],
    top_k: int,
) -> List[str]:
    ranked_scores = scores.copy()
    for jid in exclude_ids:
        idx = job_index.get(jid)
        if idx is not None:
            ranked_scores[idx] = -np.inf
    order = np.argsort(ranked_scores)[::-1]
    ranked_ids = [jobs.loc[i, "job_id"] for i in order if ranked_scores[i] != -np.inf]
    return ranked_ids[:top_k]


def _rank_candidates(
    scores: np.ndarray,
    candidate_ids: List[str],
    job_index: Dict[str, int],
    top_k: int,
) -> List[str]:
    valid_ids = [cid for cid in candidate_ids if cid in job_index]
    if not valid_ids:
        return []
    candidate_idx = [job_index[cid] for cid in valid_ids]
    candidate_scores = np.asarray(scores, dtype=float)[candidate_idx]
    order = np.argsort(candidate_scores)[::-1]
    ranked_ids = [valid_ids[i] for i in order]
    return ranked_ids[:top_k]


def _build_candidate_ids(
    jobs: pd.DataFrame,
    user_row: pd.Series,
    train_seen: Set[str],
    positives: Set[str],
    rng: np.random.Generator,
    negative_sample_size: int,
) -> List[str]:
    candidate_jobs = _filter_candidates_for_user(jobs, user_row)
    if train_seen:
        candidate_jobs = candidate_jobs[~candidate_jobs["job_id"].isin(train_seen)]
    negative_pool = [jid for jid in candidate_jobs["job_id"].tolist() if jid not in positives]

    if negative_sample_size <= 0 or not negative_pool:
        sampled_negatives = []
    else:
        n_neg = min(negative_sample_size, len(negative_pool))
        sampled_negatives = rng.choice(negative_pool, size=n_neg, replace=False).tolist()

    positives_list = sorted(positives)
    return positives_list + sampled_negatives


def _empty_metric_store(names: Iterable[str], alphas: Iterable[float], eval_ks: Iterable[int]):
    def _init_bucket():
        return {k: {"precision": [], "recall": [], "ndcg": [], "hr": [], "map": []} for k in eval_ks}

    metrics: Dict[str, Dict[int, Dict[str, List[float]]]] = {name: _init_bucket() for name in names}
    alpha_metrics: Dict[float, Dict[int, Dict[str, List[float]]]] = {alpha: _init_bucket() for alpha in alphas}
    return metrics, alpha_metrics


def _update_metric(
    store: Dict[int, Dict[str, List[float]]],
    predicted: List[str],
    actual: Set[str],
    eval_ks: Iterable[int],
) -> None:
    for k in eval_ks:
        store[k]["precision"].append(precision_at_k(predicted, actual, k))
        store[k]["recall"].append(recall_at_k(predicted, actual, k))
        store[k]["ndcg"].append(ndcg_at_k(predicted, actual, k))
        store[k]["hr"].append(hit_rate_at_k(predicted, actual, k))
        store[k]["map"].append(average_precision_at_k(predicted, actual, k))


def _build_score_registry(
    job_tfidf,
    user_tfidf,
    job_sbert: np.ndarray,
    user_sbert: np.ndarray,
    lfm_model,
    lfm_dataset,
    lfm_jobs: pd.DataFrame,
    user_features_matrix,
    item_features_matrix,
    ncf_model,
    ncf_user_index: Dict[str, int],
    ncf_job_index: Dict[str, int],
) -> Dict[str, callable]:
    """
    Register scoring functions for each model to keep evaluation declarative.
    """
    scorers = {
        "tfidf": lambda user_id, user_idx: compute_content_scores_for_user(user_idx, job_tfidf, user_tfidf),
        "sbert": lambda user_id, user_idx: compute_sbert_scores_for_user(user_idx, job_sbert, user_sbert),
        "lightfm": lambda user_id, user_idx: predict_lightfm_scores_for_user(
            user_id=user_id,
            model=lfm_model,
            dataset=lfm_dataset,
            jobs=lfm_jobs,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
        ),
        "ncf": lambda user_id, user_idx: predict_ncf_scores_for_user(
            user_id=user_id,
            model=ncf_model,
            job_index=ncf_job_index,
            user_index=ncf_user_index,
            n_items=len(lfm_jobs),
        ),
        "random": lambda user_id, user_idx, rng=None: rng.random(len(lfm_jobs)),
    }
    for name, fn in scorers.items():
        register_model(name, fn)
    return scorers


def evaluate_all_models(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    job_sbert: np.ndarray,
    user_sbert: np.ndarray,
    lfm_model,
    lfm_dataset,
    user_features_matrix,
    item_features_matrix,
    ncf_model,
    ncf_user_index: Dict[str, int],
    ncf_job_index: Dict[str, int],
    alphas: List[float],
    top_k: int,
    negative_sample_size: int,
    seed: int = EVAL_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    job_index = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    user_index = {uid: idx for idx, uid in enumerate(users["user_id"])}
    rng = np.random.default_rng(seed)
    score_rng = np.random.default_rng(seed + 1)
    train_seen_map = train_interactions.groupby("user_id")["job_id"].apply(set).to_dict()
    eval_ks = sorted(set(EVAL_KS + [top_k]))
    max_k = max(eval_ks)

    metrics, alpha_metrics = _empty_metric_store(
        names=["tfidf", "sbert", "lightfm", "ncf", "random", f"hybrid_alpha_{DEFAULT_ALPHA}"],
        alphas=alphas,
        eval_ks=eval_ks,
    )

    scorers = _build_score_registry(
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        job_sbert=job_sbert,
        user_sbert=user_sbert,
        lfm_model=lfm_model,
        lfm_dataset=lfm_dataset,
        lfm_jobs=jobs,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        ncf_model=ncf_model,
        ncf_user_index=ncf_user_index,
        ncf_job_index=ncf_job_index,
    )

    for user_id, group in test_interactions.groupby("user_id"):
        ground_truth = set(group["job_id"])
        if len(ground_truth) == 0 or user_id not in user_index:
            continue

        train_seen = train_seen_map.get(user_id, set())
        uidx = user_index[user_id]
        user_row = users.iloc[uidx]

        candidate_ids = _build_candidate_ids(
            jobs=jobs,
            user_row=user_row,
            train_seen=train_seen,
            positives=ground_truth,
            rng=rng,
            negative_sample_size=negative_sample_size,
        )
        if not candidate_ids:
            continue

        valid_candidate_ids = [cid for cid in candidate_ids if cid in job_index]
        if not valid_candidate_ids:
            continue

        tfidf_scores = scorers["tfidf"](user_id, uidx)
        sbert_scores = scorers["sbert"](user_id, uidx)
        lfm_scores = scorers["lightfm"](user_id, uidx)
        ncf_scores = scorers["ncf"](user_id, uidx)
        random_scores = scorers["random"](user_id, uidx, rng=score_rng)

        hybrid_scores_default, content_norm, lfm_norm = compute_hybrid_scores(
            content_scores=tfidf_scores,
            lfm_scores=lfm_scores,
            alpha=DEFAULT_ALPHA,
        )

        tfidf_ranked = _rank_candidates(content_norm, valid_candidate_ids, job_index, max_k)
        sbert_ranked = _rank_candidates(sbert_scores, valid_candidate_ids, job_index, max_k)
        lfm_ranked = _rank_candidates(lfm_norm, valid_candidate_ids, job_index, max_k)
        ncf_ranked = _rank_candidates(ncf_scores, valid_candidate_ids, job_index, max_k)
        hybrid_ranked = _rank_candidates(hybrid_scores_default, valid_candidate_ids, job_index, max_k)
        random_ranked = _rank_candidates(random_scores, valid_candidate_ids, job_index, max_k)

        _update_metric(metrics["tfidf"], tfidf_ranked, ground_truth, eval_ks)
        _update_metric(metrics["sbert"], sbert_ranked, ground_truth, eval_ks)
        _update_metric(metrics["lightfm"], lfm_ranked, ground_truth, eval_ks)
        _update_metric(metrics["ncf"], ncf_ranked, ground_truth, eval_ks)
        _update_metric(metrics["random"], random_ranked, ground_truth, eval_ks)
        _update_metric(metrics[f"hybrid_alpha_{DEFAULT_ALPHA}"], hybrid_ranked, ground_truth, eval_ks)

        for alpha in alphas:
            hybrid_scores = alpha * content_norm + (1 - alpha) * lfm_norm
            hybrid_ranked_alpha = _rank_candidates(hybrid_scores, valid_candidate_ids, job_index, max_k)
            _update_metric(alpha_metrics[alpha], hybrid_ranked_alpha, ground_truth, eval_ks)

    eval_rows = []
    for name, store in metrics.items():
        row = {"model": name}
        for k in eval_ks:
            row[f"precision_at_{k}"] = mean(store[k]["precision"])
            row[f"recall_at_{k}"] = mean(store[k]["recall"])
            row[f"ndcg_at_{k}"] = mean(store[k]["ndcg"])
            row[f"hr_at_{k}"] = mean(store[k]["hr"])
            row[f"map_at_{k}"] = mean(store[k]["map"])
        row["precision_at_k"] = row[f"precision_at_{top_k}"]
        row["recall_at_k"] = row[f"recall_at_{top_k}"]
        row["ndcg_at_k"] = row[f"ndcg_at_{top_k}"]
        row["hr_at_k"] = row[f"hr_at_{top_k}"]
        row["map_at_k"] = row[f"map_at_{top_k}"]
        eval_rows.append(row)

    alpha_rows = []
    for alpha, store in alpha_metrics.items():
        row = {"alpha": alpha}
        for k in eval_ks:
            row[f"precision_at_{k}"] = mean(store[k]["precision"])
            row[f"recall_at_{k}"] = mean(store[k]["recall"])
            row[f"ndcg_at_{k}"] = mean(store[k]["ndcg"])
            row[f"hr_at_{k}"] = mean(store[k]["hr"])
            row[f"map_at_{k}"] = mean(store[k]["map"])
        row["precision_at_k"] = row[f"precision_at_{top_k}"]
        row["recall_at_k"] = row[f"recall_at_{top_k}"]
        row["ndcg_at_k"] = row[f"ndcg_at_{top_k}"]
        row["hr_at_k"] = row[f"hr_at_{top_k}"]
        row["map_at_k"] = row[f"map_at_{top_k}"]
        alpha_rows.append(row)

    alpha_df = pd.DataFrame(alpha_rows).sort_values("alpha").reset_index(drop=True)
    best_alpha_row = select_best_alpha(alpha_df)
    best_row = {"model": "hybrid_best_alpha"}
    for col in alpha_df.columns:
        if col == "alpha":
            continue
        best_row[col] = best_alpha_row[col]
    eval_rows.append(best_row)

    eval_df = pd.DataFrame(eval_rows)
    return eval_df, alpha_df


def _filter_candidates_for_user(
    jobs: pd.DataFrame,
    user_row: pd.Series,
) -> pd.DataFrame:
    """
    Optional evaluation-only filtering to tighten candidate space.
    """
    if not ENABLE_EVAL_FILTERING:
        return jobs

    preferred_location = str(user_row.get("preferred_location", "")).strip().lower()
    target_role = str(user_row.get("target_role", "")).strip().lower()

    filtered = jobs
    if preferred_location:
        mask_loc = filtered["job_location"].str.lower().str.contains(preferred_location, regex=False, na=False)
        loc_filtered = filtered[mask_loc]
        if not loc_filtered.empty:
            filtered = loc_filtered

    if target_role:
        mask_role = filtered["job_category"].str.lower().str.contains(target_role, regex=False, na=False)
        role_filtered = filtered[mask_role]
        if not role_filtered.empty:
            filtered = role_filtered

    return filtered if not filtered.empty else jobs


def evaluate_thresholds(
    users: pd.DataFrame,
    jobs: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    job_tfidf,
    user_tfidf,
    job_sbert: np.ndarray,
    user_sbert: np.ndarray,
    lfm_model,
    lfm_dataset,
    user_features_matrix,
    item_features_matrix,
    ncf_model,
    ncf_user_index: Dict[str, int],
    ncf_job_index: Dict[str, int],
    thresholds: List[float],
    negative_sample_size: int,
    alpha: float,
    seed: int = EVAL_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    job_index = {jid: idx for idx, jid in enumerate(jobs["job_id"])}
    user_index = {uid: idx for idx, uid in enumerate(users["user_id"])}
    rng = np.random.default_rng(seed)
    train_seen_map = train_interactions.groupby("user_id")["job_id"].apply(set).to_dict()

    rows = []
    candidate_sizes: List[int] = []

    for user_id, group in test_interactions.groupby("user_id"):
        ground_truth = set(group["job_id"])
        if len(ground_truth) == 0 or user_id not in user_index:
            continue

        train_seen = train_seen_map.get(user_id, set())
        uidx = user_index[user_id]
        user_row = users.iloc[uidx]

        tfidf_scores = compute_content_scores_for_user(uidx, job_tfidf, user_tfidf)
        sbert_scores = compute_sbert_scores_for_user(uidx, job_sbert, user_sbert)
        lfm_scores = predict_lightfm_scores_for_user(
            user_id=user_id,
            model=lfm_model,
            dataset=lfm_dataset,
            jobs=jobs,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
        )
        ncf_scores = predict_ncf_scores_for_user(
            user_id=user_id,
            model=ncf_model,
            job_index=ncf_job_index,
            user_index=ncf_user_index,
            n_items=len(jobs),
        )
        random_scores = rng.random(len(jobs))

        hybrid_scores, content_norm, lfm_norm = compute_hybrid_scores(
            content_scores=tfidf_scores,
            lfm_scores=lfm_scores,
            alpha=alpha,
        )

        candidate_ids = _build_candidate_ids(
            jobs=jobs,
            user_row=user_row,
            train_seen=train_seen,
            positives=ground_truth,
            rng=rng,
            negative_sample_size=negative_sample_size,
        )
        candidate_sizes.append(len(candidate_ids))
        if not candidate_ids:
            continue

        candidate_ids = [cid for cid in candidate_ids if cid in job_index]
        candidate_idx = [job_index[cid] for cid in candidate_ids]
        if not candidate_idx:
            continue

        positive_mask = np.array([cid in ground_truth for cid in candidate_ids], dtype=bool)
        n_pos = int(positive_mask.sum())
        n_neg = len(candidate_ids) - n_pos

        model_scores = {
            "tfidf": tfidf_scores[candidate_idx],
            "sbert": sbert_scores[candidate_idx],
            "lightfm": lfm_scores[candidate_idx],
            "ncf": ncf_scores[candidate_idx],
            f"hybrid_alpha_{alpha}": hybrid_scores[candidate_idx],
            "random": random_scores[candidate_idx],
        }

        for model_name, scores in model_scores.items():
            norm_scores = minmax_normalize(np.array(scores, dtype=float))
            for t in thresholds:
                preds = norm_scores >= t
                tp = int(np.logical_and(preds, positive_mask).sum())
                fp = int((preds & ~positive_mask).sum())
                fn = int((~preds & positive_mask).sum())
                precision, recall, f1 = precision_recall_f1(tp, fp, fn)

                rows.append(
                    {
                        "user_id": user_id,
                        "model": model_name,
                        "threshold": t,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "n_pos": n_pos,
                        "n_neg": n_neg,
                    }
                )

    if ENABLE_EVAL_FILTERING and candidate_sizes:
        avg_size = sum(candidate_sizes) / len(candidate_sizes)
        print(f"[Eval filtering ON] Avg candidate size per user: {avg_size:.2f}")

    results_df = pd.DataFrame(rows)
    summary_df = (
        results_df.groupby(["model", "threshold"])[["precision", "recall", "f1"]]
        .mean()
        .reset_index()
        .rename(columns={"precision": "mean_precision", "recall": "mean_recall", "f1": "mean_f1"})
    )
    return results_df, summary_df


def select_best_alpha(alpha_df: pd.DataFrame) -> pd.Series:
    sort_cols = ["ndcg_at_k", "recall_at_k", "precision_at_k"]
    return alpha_df.sort_values(sort_cols, ascending=False).iloc[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommender models.")
    parser.add_argument(
        "--negative-sample-size",
        type=int,
        default=NEGATIVE_SAMPLE_SIZE,
        help="Number of sampled negatives per user for ranking evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test split size per user when not using leave-one-out.",
    )
    parser.add_argument(
        "--leave-one-out",
        dest="leave_one_out",
        action="store_true",
        help="Hold out exactly one positive per user for evaluation.",
    )
    parser.add_argument(
        "--no-leave-one-out",
        dest="leave_one_out",
        action="store_false",
        help="Disable leave-one-out and use test-size split instead.",
    )
    parser.add_argument(
        "--eval-filtering",
        dest="eval_filtering",
        action="store_true",
        help="Filter candidate jobs by preferred location/target role during evaluation.",
    )
    parser.add_argument(
        "--no-eval-filtering",
        dest="eval_filtering",
        action="store_false",
        help="Disable candidate filtering during evaluation.",
    )
    parser.set_defaults(leave_one_out=LEAVE_ONE_OUT)
    parser.set_defaults(eval_filtering=ENABLE_EVAL_FILTERING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    negative_sample_size = max(0, int(args.negative_sample_size))
    test_size = float(args.test_size)
    leave_one_out = bool(args.leave_one_out)
    global ENABLE_EVAL_FILTERING
    ENABLE_EVAL_FILTERING = bool(args.eval_filtering)
    set_seed(EVAL_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_jobs = load_jobstreet_job_dataset()
    raw_applicants = load_applicants_dataset()

    if len(raw_jobs) > MAX_JOBS:
        raw_jobs = raw_jobs.sample(MAX_JOBS, random_state=42)
    if len(raw_applicants) > MAX_USERS:
        raw_applicants = raw_applicants.sample(MAX_USERS, random_state=42)

    jobs = build_job_table(raw_jobs).reset_index(drop=True)
    users = build_user_table(raw_applicants).reset_index(drop=True)
    print(f"Prepared {len(jobs)} jobs and {len(users)} users.")

    interactions_df = build_synthetic_interactions(users, jobs, seed=EVAL_SEED)
    train_interactions, test_interactions = train_test_split_interactions(
        interactions_df,
        test_size=test_size,
        seed=EVAL_SEED,
        leave_one_out=leave_one_out,
    )
    print(f"Train interactions: {len(train_interactions)}, Test interactions: {len(test_interactions)}")
    if leave_one_out:
        split_note = "leave-one-out"
    else:
        split_note = f"test_size={test_size:.2f}"
    filtering_note = "on" if ENABLE_EVAL_FILTERING else "off"
    k_note = ",".join(str(k) for k in sorted(set(EVAL_KS + [TOP_K])))
    print(
        f"Evaluation protocol: implicit feedback with negative sampling "
        f"(N={negative_sample_size}), split={split_note}, filtering={filtering_note}, ks={k_note}"
    )

    # Content-based models
    _, job_tfidf, user_tfidf = build_tfidf_representations(users, jobs)
    _, job_sbert, user_sbert = build_sbert_representations(
        users=users["user_text"].values,
        jobs=jobs["job_text"].values,
        seed=EVAL_SEED,
    )

    # LightFM
    dataset, interactions, weights, user_features_matrix, item_features_matrix = build_lightfm_dataset(
        interactions_df=train_interactions,
        users=users,
        jobs=jobs,
    )
    lfm_model = train_lightfm(
        interactions=interactions,
        weights=weights,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )

    # NCF
    (
        ncf_user_indices,
        ncf_item_indices,
        ncf_labels,
        ncf_user_index,
        ncf_job_index,
    ) = build_ncf_training_data(train_interactions, users, jobs)
    if len(ncf_labels) == 0:
        raise ValueError("NCF training data is empty. Check interactions preprocessing.")
    ncf_model = train_ncf_model(
        user_indices=ncf_user_indices,
        item_indices=ncf_item_indices,
        labels=ncf_labels,
        n_users=len(users),
        n_items=len(jobs),
        epochs=4,
        embedding_dim=32,
    )

    alpha_candidates = sorted(set(ALPHAS + [DEFAULT_ALPHA]))
    eval_df, alpha_df = evaluate_all_models(
        users=users,
        jobs=jobs,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        job_sbert=job_sbert,
        user_sbert=user_sbert,
        lfm_model=lfm_model,
        lfm_dataset=dataset,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        ncf_model=ncf_model,
        ncf_user_index=ncf_user_index,
        ncf_job_index=ncf_job_index,
        alphas=alpha_candidates,
        top_k=TOP_K,
        negative_sample_size=negative_sample_size,
        seed=EVAL_SEED,
    )

    eval_path = OUT_DIR / "evaluation_results.csv"
    alpha_path = OUT_DIR / "alpha_tuning_results.csv"
    eval_df.to_csv(eval_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)

    best_alpha = select_best_alpha(alpha_df)["alpha"]
    threshold_results, threshold_summary = evaluate_thresholds(
        users=users,
        jobs=jobs,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        job_tfidf=job_tfidf,
        user_tfidf=user_tfidf,
        job_sbert=job_sbert,
        user_sbert=user_sbert,
        lfm_model=lfm_model,
        lfm_dataset=dataset,
        user_features_matrix=user_features_matrix,
        item_features_matrix=item_features_matrix,
        ncf_model=ncf_model,
        ncf_user_index=ncf_user_index,
        ncf_job_index=ncf_job_index,
        thresholds=THRESHOLDS,
        negative_sample_size=negative_sample_size,
        alpha=DEFAULT_ALPHA,
        seed=EVAL_SEED,
    )

    threshold_path = OUT_DIR / "threshold_results.csv"
    threshold_summary_path = OUT_DIR / "threshold_results_summary.csv"
    threshold_results.to_csv(threshold_path, index=False)
    threshold_summary.to_csv(threshold_summary_path, index=False)

    print("Saved evaluation outputs:")
    print(f"- {eval_path}")
    print(f"- {alpha_path}")
    print(f"- {threshold_path}")
    print(f"- {threshold_summary_path}")
    print(f"Best alpha (by NDCG@{TOP_K}, recall tie-breaker): {best_alpha}")


if __name__ == "__main__":
    main()

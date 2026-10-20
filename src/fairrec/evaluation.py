"""Leakage-safe ranking and representation-fairness evaluation.

The functions in this module deliberately accept arrays rather than a concrete
model or data-loader class.  This keeps the evaluation protocol shared by
Popularity, ItemKNN, MultVAE, and adversarial MultVAE:

* a user is either in train, validation, or test (never more than one);
* a cold user's first N train-warm positive events form support and later
  positives (rating >= 4 by default) are relevant;
* support and train-cold items are removed from ranking with ``-inf``;
* fairness is reported as a measurement, not inferred from accuracy parity.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GENDER_NAMES = {0: "male", 1: "female"}


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return array


def _gender_code(value: Any) -> int | None:
    """Map MovieLens and human-readable gender values to 0/1."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"m", "male", "0"}:
            return 0
        if normalized in {"f", "female", "1"}:
            return 1
        return None
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        integer = int(value)
        return integer if integer in (0, 1) else None
    return None


def normalize_genders(genders: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return int8 gender codes, using -1 for unsupported/missing values."""
    flat = np.asarray(genders).reshape(-1)
    return np.asarray(
        [-1 if _gender_code(value) is None else _gender_code(value) for value in flat],
        dtype=np.int8,
    )


def mask_seen_scores(scores: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Copy scores and set all observed items to negative infinity.

    ``seen`` is an aligned boolean/0-1 matrix containing the positive movies
    supplied to the implicit recommender as its support history.
    """
    score_array = _as_2d("scores", np.asarray(scores, dtype=np.float64))
    seen_array = _as_2d("seen", np.asarray(seen, dtype=bool))
    if score_array.shape != seen_array.shape:
        raise ValueError("scores and seen must have identical shapes")
    masked = score_array.copy()
    masked[seen_array] = -np.inf
    return masked


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Select finite Top-K indices with item id as the deterministic tie-break."""
    row = np.asarray(scores, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k <= 0:
        raise ValueError("k must be positive")
    candidates = np.flatnonzero(np.isfinite(row))
    if candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((candidates, -row[candidates]))
    return candidates[order[: min(k, candidates.size)]].astype(np.int64, copy=False)


def ranking_metrics_at_k(
    scores: np.ndarray,
    relevant: np.ndarray,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Compute binary Recall, NDCG, and HitRate against a full catalog."""
    row = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(relevant, dtype=bool)
    if row.ndim != 1 or truth.ndim != 1 or row.shape != truth.shape:
        raise ValueError("scores and relevant must be aligned vectors")
    relevant_count = int(truth.sum())
    if relevant_count == 0:
        raise ValueError("ranking metrics require at least one relevant item")
    recommended = stable_topk(row, k)
    hits = truth[recommended].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(hits.size, dtype=np.float64) + 2.0)
    dcg = float(np.dot(hits, discounts))
    ideal_count = min(k, relevant_count)
    ideal_discounts = 1.0 / np.log2(
        np.arange(ideal_count, dtype=np.float64) + 2.0
    )
    return {
        f"ndcg@{k}": dcg / float(ideal_discounts.sum()),
        f"recall@{k}": float(hits.sum() / relevant_count),
        f"hitrate@{k}": float(bool(hits.any())),
    }


def _mean_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }


def _genre_distribution(
    item_rows: Sequence[np.ndarray],
    movie_genres: np.ndarray,
) -> np.ndarray:
    totals = np.zeros(movie_genres.shape[1], dtype=np.float64)
    for items in item_rows:
        indices = np.asarray(items, dtype=np.int64)
        if indices.size:
            totals += movie_genres[indices].sum(axis=0, dtype=np.float64)
    denominator = float(totals.sum())
    return totals / denominator if denominator > 0.0 else totals


def _history_genre_distribution(
    histories: np.ndarray,
    user_mask: np.ndarray,
    movie_genres: np.ndarray,
) -> np.ndarray:
    item_counts = np.asarray(histories[user_mask], dtype=np.float64).sum(axis=0)
    totals = item_counts @ movie_genres.astype(np.float64, copy=False)
    denominator = float(totals.sum())
    return totals / denominator if denominator > 0.0 else totals


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)

    def divergence(part: np.ndarray) -> float:
        valid = (part > 0.0) & (middle > 0.0)
        return float(np.sum(part[valid] * np.log2(part[valid] / middle[valid])))

    return 0.5 * divergence(left) + 0.5 * divergence(right)


def genre_exposure_diagnostics(
    recommendations: Sequence[np.ndarray],
    genders: Sequence[Any] | np.ndarray,
    movie_genres: np.ndarray,
    *,
    genre_names: Sequence[str] | None = None,
    positive_histories: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure hard recommendation exposure and preference amplification.

    A multi-genre movie contributes one count to every listed genre.  Counts
    are normalized into a distribution before L1 and Jensen-Shannon distances
    are computed.  Preference amplification is recommendation disparity minus
    the disparity in users' observed positive histories; it is diagnostic, not
    a causal estimate.
    """
    features = _as_2d("movie_genres", np.asarray(movie_genres, dtype=np.float64))
    if len(recommendations) != len(genders):
        raise ValueError("recommendations and genders must have equal length")
    names = (
        [str(name) for name in genre_names]
        if genre_names is not None
        else [f"genre_{index}" for index in range(features.shape[1])]
    )
    if len(names) != features.shape[1]:
        raise ValueError("genre_names length must match movie_genres width")
    codes = normalize_genders(genders)
    distributions: dict[int, np.ndarray] = {}
    for code in (0, 1):
        rows = [
            np.asarray(items, dtype=np.int64)
            for items, row_code in zip(recommendations, codes, strict=True)
            if row_code == code
        ]
        distributions[code] = _genre_distribution(rows, features)

    male = distributions[0]
    female = distributions[1]
    signed_recommendation_gap = female - male
    result: dict[str, Any] = {
        "normalization": "multi-label genre tag share over hard Top-K items",
        "distribution_by_gender": {
            GENDER_NAMES[code]: {
                name: float(value)
                for name, value in zip(names, distributions[code], strict=True)
            }
            for code in (0, 1)
        },
        "female_minus_male_by_genre": {
            name: float(value)
            for name, value in zip(names, signed_recommendation_gap, strict=True)
        },
        "l1_between_genders": float(np.abs(signed_recommendation_gap).sum()),
        "jensen_shannon_between_genders": _jensen_shannon(male, female),
    }

    if positive_histories is not None:
        histories = _as_2d(
            "positive_histories", np.asarray(positive_histories, dtype=np.float64)
        )
        if histories.shape != (len(codes), features.shape[0]):
            raise ValueError(
                "positive_histories must have shape [users, number of movies]"
            )
        history_distributions = {
            code: _history_genre_distribution(histories, codes == code, features)
            for code in (0, 1)
        }
        signed_history_gap = history_distributions[1] - history_distributions[0]
        history_l1 = float(np.abs(signed_history_gap).sum())
        recommendation_l1 = result["l1_between_genders"]
        residual = signed_recommendation_gap - signed_history_gap
        result["preference_amplification"] = {
            "interpretation": (
                "recommendation gender disparity minus observed-positive-history "
                "gender disparity; descriptive, not causal"
            ),
            "history_l1_between_genders": history_l1,
            "recommendation_l1_between_genders": recommendation_l1,
            "scalar_l1_amplification": recommendation_l1 - history_l1,
            "residual_l1": float(np.abs(residual).sum()),
            "female_minus_male_history_by_genre": {
                name: float(value)
                for name, value in zip(names, signed_history_gap, strict=True)
            },
            "amplification_by_genre": {
                name: float(value)
                for name, value in zip(names, residual, strict=True)
            },
        }
    return result


def evaluate_score_matrix(
    scores: np.ndarray,
    query_positive: np.ndarray,
    seen: np.ndarray,
    genders: Sequence[Any] | np.ndarray,
    *,
    k: int = 10,
    movie_genres: np.ndarray | None = None,
    genre_names: Sequence[str] | None = None,
    positive_histories: np.ndarray | None = None,
    catalog_eligible: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate full-catalog scores for cold-start users.

    Users without a positive future-query item are reported and omitted from
    ranking means, because Recall/NDCG have an undefined denominator for them.
    Exposure and coverage use every known-gender support profile; ranking means
    separately omit users whose future query has no eligible positive item.
    """
    score_array = _as_2d("scores", np.asarray(scores, dtype=np.float64))
    truth = _as_2d("query_positive", np.asarray(query_positive, dtype=bool))
    seen_array = _as_2d("seen", np.asarray(seen, dtype=bool))
    if score_array.shape != truth.shape or score_array.shape != seen_array.shape:
        raise ValueError("scores, query_positive, and seen must have equal shapes")
    codes = normalize_genders(genders)
    if codes.size != score_array.shape[0]:
        raise ValueError("one gender value is required per score row")

    item_eligible = (
        np.ones(score_array.shape[1], dtype=bool)
        if catalog_eligible is None
        else np.asarray(catalog_eligible, dtype=bool).reshape(-1)
    )
    if item_eligible.shape != (score_array.shape[1],):
        raise ValueError("catalog_eligible must have one value per catalog item")
    if not item_eligible.any():
        raise ValueError("catalog_eligible must contain at least one warm item")
    masked = mask_seen_scores(score_array, seen_array)
    masked[:, ~item_eligible] = -np.inf
    # A query event cannot be relevant if it was already consumed as support.
    cold_query_positives = truth & ~item_eligible[None, :]
    effective_truth = truth & ~seen_array & item_eligible[None, :]
    eligible = effective_truth.any(axis=1)
    rows: list[dict[str, float]] = []
    rows_by_gender: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    recommendations: list[np.ndarray] = []
    recommendation_genders: list[int] = []
    eligible_indices = np.flatnonzero(eligible)
    for row_index in eligible_indices:
        metrics = ranking_metrics_at_k(
            masked[row_index], effective_truth[row_index], k=k
        )
        rows.append(metrics)
        code = int(codes[row_index])
        if code in rows_by_gender:
            rows_by_gender[code].append(metrics)
    exposure_indices = np.flatnonzero(np.isin(codes, [0, 1]))
    for row_index in exposure_indices:
        recommendations.append(stable_topk(masked[row_index], k))
        recommendation_genders.append(int(codes[row_index]))

    overall = _mean_rows(rows)
    by_gender = {
        GENDER_NAMES[code]: {
            **_mean_rows(rows_by_gender[code]),
            "users": len(rows_by_gender[code]),
        }
        for code in (0, 1)
    }
    metric_keys = (f"ndcg@{k}", f"recall@{k}", f"hitrate@{k}")
    gaps = {
        key: abs(by_gender["female"].get(key, math.nan) - by_gender["male"].get(key, math.nan))
        for key in metric_keys
    }
    unique_recommended = {
        int(item) for row in recommendations for item in np.asarray(row).tolist()
    }
    result: dict[str, Any] = {
        **overall,
        "metrics_by_gender": by_gender,
        "absolute_gender_gaps": gaps,
        "users_total": int(score_array.shape[0]),
        "users_evaluated": int(eligible.sum()),
        "users_without_future_positive": int((~eligible).sum()),
        "users_with_known_binary_gender": int(np.isin(codes, [0, 1]).sum()),
        "cohort_by_gender": {
            GENDER_NAMES[code]: {
                "total": int(np.count_nonzero(codes == code)),
                "ranking_evaluated": int(np.count_nonzero((codes == code) & eligible)),
                "without_future_positive": int(np.count_nonzero((codes == code) & ~eligible)),
                "ranking_eligibility_rate": float(
                    np.mean(eligible[codes == code]) if np.any(codes == code) else math.nan
                ),
            }
            for code in (0, 1)
        },
        "coverage": float(len(unique_recommended) / int(item_eligible.sum())),
        "unique_recommended_items": len(unique_recommended),
        "catalog_items": int(score_array.shape[1]),
        "eligible_catalog_items": int(item_eligible.sum()),
        "ineligible_catalog_items": int((~item_eligible).sum()),
        "excluded_cold_query_positives": int(cold_query_positives.sum()),
        "users_with_only_cold_future_positives": int(
            np.count_nonzero(truth.any(axis=1) & ~effective_truth.any(axis=1))
        ),
    }
    if movie_genres is not None:
        histories = (
            (np.asarray(positive_histories) * item_eligible[None, :])[
                exposure_indices
            ]
            if positive_histories is not None
            else None
        )
        result["genre_exposure"] = genre_exposure_diagnostics(
            recommendations,
            recommendation_genders,
            movie_genres,
            genre_names=genre_names,
            positive_histories=histories,
        )
    return result


def binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Tie-aware ROC AUC without an optional scikit-learn dependency."""
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    score = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.shape != score.shape:
        raise ValueError("labels and probabilities must be aligned")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        end = start + 1
        while end < y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def balanced_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    prediction = np.asarray(probabilities, dtype=np.float64).reshape(-1) >= threshold
    if y.shape != prediction.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("aligned binary labels are required")
    recalls: list[float] = []
    for label in (0, 1):
        selected = y == label
        if not selected.any():
            return math.nan
        recalls.append(float(np.mean(prediction[selected] == bool(label))))
    return float(np.mean(recalls))


def _best_balanced_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    unique = np.unique(np.asarray(probabilities, dtype=np.float64))
    if unique.size == 1:
        candidates = np.asarray([unique[0]], dtype=np.float64)
    else:
        candidates = np.concatenate(
            ([np.nextafter(unique[0], -np.inf)], 0.5 * (unique[:-1] + unique[1:]), [np.nextafter(unique[-1], np.inf)])
        )
    scored = [balanced_accuracy(labels, probabilities, threshold=float(value)) for value in candidates]
    # Prefer a threshold closest to 0.5 when validation values tie.
    best = max(
        range(len(candidates)),
        key=lambda index: (scored[index], -abs(float(candidates[index]) - 0.5)),
    )
    return float(candidates[best])


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    positive = logits >= 0.0
    output = np.empty_like(logits, dtype=np.float64)
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass(frozen=True)
class _ProbeModel:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return _sigmoid(normalized @ self.weight + self.bias)


def _train_logistic_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    seed: int,
    steps: int,
    learning_rate: float,
) -> _ProbeModel:
    x = _as_2d("features", np.asarray(features, dtype=np.float64))
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.size:
        raise ValueError("one label is required per feature row")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 1e-3, size=x.shape[1])
    bias = 0.0
    first_weight = np.zeros_like(weight)
    second_weight = np.zeros_like(weight)
    first_bias = 0.0
    second_bias = 0.0
    class_counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("probe training split must contain both genders")
    sample_weight = y.size / (2.0 * class_counts[y.astype(np.int64)])
    for step in range(1, steps + 1):
        probability = _sigmoid(normalized @ weight + bias)
        residual = (probability - y) * sample_weight
        grad_weight = normalized.T @ residual / y.size + l2 * weight
        grad_bias = float(residual.mean())
        first_weight = 0.9 * first_weight + 0.1 * grad_weight
        second_weight = 0.999 * second_weight + 0.001 * np.square(grad_weight)
        first_bias = 0.9 * first_bias + 0.1 * grad_bias
        second_bias = 0.999 * second_bias + 0.001 * grad_bias * grad_bias
        correction1 = 1.0 - 0.9**step
        correction2 = 1.0 - 0.999**step
        weight -= learning_rate * (first_weight / correction1) / (
            np.sqrt(second_weight / correction2) + 1e-8
        )
        bias -= learning_rate * (first_bias / correction1) / (
            math.sqrt(second_bias / correction2) + 1e-8
        )
    return _ProbeModel(mean, scale, weight, bias)


def fit_gender_probe(
    train_latents: np.ndarray,
    train_genders: Sequence[Any] | np.ndarray,
    val_latents: np.ndarray,
    val_genders: Sequence[Any] | np.ndarray,
    test_latents: np.ndarray,
    test_genders: Sequence[Any] | np.ndarray,
    *,
    l2_values: Sequence[float] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1),
    seed: int = 42,
    steps: int = 400,
    learning_rate: float = 0.03,
) -> dict[str, Any]:
    """Train on train latents, select on validation, and report test once.

    The reported AUC measures gender information recoverable by this fixed
    linear attack.  Fair representation values are close to 0.5.  No test
    representation or label participates in scaling, fitting, hyperparameter
    selection, or threshold selection.
    """
    arrays = [
        _as_2d("train_latents", np.asarray(train_latents, dtype=np.float64)),
        _as_2d("val_latents", np.asarray(val_latents, dtype=np.float64)),
        _as_2d("test_latents", np.asarray(test_latents, dtype=np.float64)),
    ]
    labels = [
        normalize_genders(train_genders),
        normalize_genders(val_genders),
        normalize_genders(test_genders),
    ]
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError("all latent splits must have the same width")
    filtered_arrays: list[np.ndarray] = []
    filtered_labels: list[np.ndarray] = []
    for name, array, split_labels in zip(
        ("train", "validation", "test"), arrays, labels, strict=True
    ):
        if array.shape[0] != split_labels.size:
            raise ValueError(f"{name} latent rows and genders differ")
        known = np.isin(split_labels, [0, 1])
        x = array[known]
        y = split_labels[known]
        if x.shape[0] == 0 or np.unique(y).size != 2:
            raise ValueError(f"{name} split needs examples from both genders")
        filtered_arrays.append(x)
        filtered_labels.append(y)
    train_x, val_x, test_x = filtered_arrays
    train_y, val_y, test_y = filtered_labels

    candidates: list[tuple[float, float, float, float, _ProbeModel, float, float]] = []
    for index, l2 in enumerate(l2_values):
        if l2 < 0:
            raise ValueError("probe L2 values must be non-negative")
        model = _train_logistic_probe(
            train_x,
            train_y,
            l2=float(l2),
            seed=seed,
            steps=steps,
            learning_rate=learning_rate,
        )
        raw_validation_probability = model.predict(val_x)
        raw_auc = binary_auc(val_y, raw_validation_probability)
        orientation = 1.0 if raw_auc >= 0.5 else -1.0
        validation_probability = (
            raw_validation_probability
            if orientation > 0
            else 1.0 - raw_validation_probability
        )
        threshold = _best_balanced_threshold(val_y, validation_probability)
        oriented_auc = binary_auc(val_y, validation_probability)
        accuracy = balanced_accuracy(
            val_y, validation_probability, threshold=threshold
        )
        candidates.append(
            (oriented_auc - 0.5, accuracy, -float(l2), raw_auc, model, threshold, orientation)
        )
    selected = max(candidates, key=lambda row: row[:3])
    _, validation_accuracy, negative_l2, raw_validation_auc, model, threshold, orientation = selected
    raw_test_probability = model.predict(test_x)
    test_probability = (
        raw_test_probability if orientation > 0 else 1.0 - raw_test_probability
    )
    validation_auc = max(raw_validation_auc, 1.0 - raw_validation_auc)
    return {
        "protocol": {
            "fit": "train users only",
            "model_selection_and_threshold": "validation users only",
            "report": "test users only",
            "classifier": "class-balanced linear logistic regression (limited linear attack)",
            "selection": "strongest validation abs(AUC - 0.5); direction invariant",
            "positive_class": "female after validation-only orientation",
        },
        "selected_l2": -negative_l2,
        "selected_threshold": threshold,
        "validation": {
            "auc": validation_auc,
            "auc_distance_from_random": abs(validation_auc - 0.5),
            "balanced_accuracy": validation_accuracy,
            "users": int(val_y.size),
        },
        "test": {
            "auc": binary_auc(test_y, test_probability),
            "auc_distance_from_random": abs(binary_auc(test_y, test_probability) - 0.5),
            "balanced_accuracy": balanced_accuracy(
                test_y, test_probability, threshold=threshold
            ),
            "users": int(test_y.size),
        },
        "train_users": int(train_y.size),
    }


class PopularityRecommender:
    """Non-personalized positive-interaction popularity baseline."""

    def __init__(self) -> None:
        self.scores_: np.ndarray | None = None

    def fit(self, train_positive: np.ndarray) -> "PopularityRecommender":
        matrix = _as_2d("train_positive", np.asarray(train_positive, dtype=np.float64))
        self.scores_ = matrix.sum(axis=0, dtype=np.float64)
        return self

    def score(self, support_positive: np.ndarray) -> np.ndarray:
        support = _as_2d("support_positive", np.asarray(support_positive))
        if self.scores_ is None:
            raise RuntimeError("PopularityRecommender must be fit before score")
        if support.shape[1] != self.scores_.size:
            raise ValueError("support item count differs from fitted catalog")
        return np.broadcast_to(self.scores_, support.shape).copy()


class ItemKNNRecommender:
    """Implicit cosine ItemKNN trained solely on training-user positives."""

    def __init__(self, neighbors: int = 200, shrinkage: float = 10.0) -> None:
        if neighbors <= 0 or shrinkage < 0:
            raise ValueError("neighbors must be positive and shrinkage non-negative")
        self.neighbors = int(neighbors)
        self.shrinkage = float(shrinkage)
        self.similarity_: np.ndarray | None = None

    def fit(self, train_positive: np.ndarray) -> "ItemKNNRecommender":
        matrix = _as_2d(
            "train_positive", np.asarray(train_positive, dtype=np.float32)
        )
        counts = matrix.sum(axis=0, dtype=np.float64)
        norms = np.sqrt(np.maximum(counts, 1.0))
        similarity = np.asarray(matrix.T @ matrix, dtype=np.float32)
        similarity /= np.asarray(norms[:, None] * norms[None, :], dtype=np.float32)
        if self.shrinkage:
            # Cosine shrinkage based on the geometric support of each pair.
            support = np.sqrt(np.maximum(counts[:, None] * counts[None, :], 0.0))
            similarity *= np.asarray(
                support / (support + self.shrinkage), dtype=np.float32
            )
        np.fill_diagonal(similarity, 0.0)
        keep = min(self.neighbors, max(similarity.shape[0] - 1, 0))
        if keep < similarity.shape[0]:
            for item in range(similarity.shape[0]):
                row = similarity[item]
                discard = np.argpartition(row, -keep)[:-keep] if keep else np.arange(row.size)
                row[discard] = 0.0
        self.similarity_ = similarity
        return self

    def score(self, support_positive: np.ndarray) -> np.ndarray:
        support = _as_2d(
            "support_positive", np.asarray(support_positive, dtype=np.float32)
        )
        if self.similarity_ is None:
            raise RuntimeError("ItemKNNRecommender must be fit before score")
        if support.shape[1] != self.similarity_.shape[0]:
            raise ValueError("support item count differs from fitted catalog")
        return np.asarray(support @ self.similarity_, dtype=np.float64)


def json_ready(value: Any) -> Any:
    """Convert NumPy/path values and non-finite floats to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_ready(part) for key, part in sorted(value.items(), key=lambda row: str(row[0]))}
    if isinstance(value, (list, tuple)):
        return [json_ready(part) for part in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic, standards-compliant JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def pareto_front(
    candidates: Sequence[Mapping[str, Any]],
    *,
    utility_key: str = "ndcg@10",
    fairness_key: str = "probe_auc_distance",
) -> list[int]:
    """Return nondominated candidate indices (maximize utility, minimize unfairness)."""
    front: list[int] = []
    for index, candidate in enumerate(candidates):
        utility = float(candidate[utility_key])
        unfairness = float(candidate[fairness_key])
        dominated = False
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_utility = float(other[utility_key])
            other_unfairness = float(other[fairness_key])
            if (
                other_utility >= utility
                and other_unfairness <= unfairness
                and (other_utility > utility or other_unfairness < unfairness)
            ):
                dominated = True
                break
        if not dominated:
            front.append(index)
    return front


def select_pareto_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    utility_key: str = "ndcg@10",
    fairness_key: str = "probe_auc_distance",
    utility_tolerance: float = 0.01,
) -> int:
    """Select the fairest Pareto point within a relative utility tolerance.

    Selection is validation-only.  The candidate must retain at least
    ``(1-utility_tolerance)`` of the best validation utility, after which lower
    unfairness wins; utility and original order break ties deterministically.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not 0.0 <= utility_tolerance < 1.0:
        raise ValueError("utility_tolerance must be in [0, 1)")
    front = pareto_front(
        candidates, utility_key=utility_key, fairness_key=fairness_key
    )
    best_utility = max(float(row[utility_key]) for row in candidates)
    floor = best_utility * (1.0 - utility_tolerance)
    feasible = [
        index for index in front if float(candidates[index][utility_key]) >= floor
    ]
    if not feasible:
        feasible = front
    return min(
        feasible,
        key=lambda index: (
            float(candidates[index][fairness_key]),
            -float(candidates[index][utility_key]),
            index,
        ),
    )


def select_output_fair_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    utility_key: str = "ndcg@10",
    utility_gap_key: str = "ndcg_gender_gap",
    exposure_key: str = "genre_exposure_l1",
    exposure_residual_key: str = "exposure_residual_l1",
    probe_key: str = "probe_auc_distance",
    utility_tolerance: float = 0.01,
) -> int:
    """Choose output parity within a validation-only utility constraint.

    First retain candidates within ``utility_tolerance`` of the best utility.
    The deterministic order then minimizes the gender utility gap, the genre
    exposure residual relative to observed support preferences, and finally
    representation leakage.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    if not 0.0 <= utility_tolerance < 1.0:
        raise ValueError("utility_tolerance must be in [0, 1)")
    best_utility = max(float(row[utility_key]) for row in candidates)
    floor = best_utility * (1.0 - utility_tolerance)
    feasible = [
        index
        for index, row in enumerate(candidates)
        if float(row[utility_key]) >= floor
    ]
    return min(
        feasible,
        key=lambda index: (
            float(candidates[index][utility_gap_key]),
            float(candidates[index][exposure_key]),
            float(candidates[index][exposure_residual_key]),
            float(candidates[index][probe_key]),
            -float(candidates[index][utility_key]),
            index,
        ),
    )

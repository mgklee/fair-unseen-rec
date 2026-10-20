from __future__ import annotations

import json

import numpy as np
import pytest

from fairrec.evaluation import (
    ItemKNNRecommender,
    PopularityRecommender,
    balanced_accuracy,
    binary_auc,
    evaluate_score_matrix,
    fit_gender_probe,
    genre_exposure_diagnostics,
    mask_seen_scores,
    pareto_front,
    ranking_metrics_at_k,
    select_output_fair_candidate,
    select_pareto_candidate,
    stable_topk,
    write_json,
)


def test_seen_items_are_negative_infinity_and_never_recommended() -> None:
    scores = np.asarray([[100.0, 4.0, 3.0], [1.0, 99.0, 2.0]])
    seen = np.asarray([[True, False, False], [False, True, False]])
    masked = mask_seen_scores(scores, seen)
    assert np.isneginf(masked[0, 0])
    assert np.isneginf(masked[1, 1])
    assert 0 not in stable_topk(masked[0], 2)
    assert 1 not in stable_topk(masked[1], 2)


def test_stable_topk_breaks_ties_by_item_index() -> None:
    assert stable_topk(np.asarray([2.0, 3.0, 3.0, 1.0]), 2).tolist() == [1, 2]


def test_ranking_metrics_use_full_catalog_and_binary_relevance() -> None:
    metrics = ranking_metrics_at_k(
        np.asarray([0.1, 0.9, 0.8, 0.7]),
        np.asarray([False, True, False, True]),
        k=2,
    )
    assert metrics["recall@2"] == pytest.approx(0.5)
    assert metrics["hitrate@2"] == 1.0
    assert metrics["ndcg@2"] == pytest.approx(1.0 / (1.0 + 1.0 / np.log2(3.0)))


def test_evaluate_score_matrix_reports_gender_gaps_and_coverage() -> None:
    scores = np.asarray(
        [[10, 9, 8, 7], [10, 9, 8, 7], [7, 8, 9, 10], [7, 8, 9, 10]],
        dtype=float,
    )
    truth = np.asarray(
        [
            [False, True, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, True, False],
        ]
    )
    seen = np.asarray(
        [
            [True, False, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, False, True],
        ]
    )
    result = evaluate_score_matrix(scores, truth, seen, [0, 0, 1, 1], k=1)
    assert result["ndcg@1"] == 1.0
    assert result["metrics_by_gender"]["male"]["recall@1"] == 1.0
    assert result["metrics_by_gender"]["female"]["recall@1"] == 1.0
    assert result["absolute_gender_gaps"]["recall@1"] == 0.0
    assert result["coverage"] == 0.5


def test_users_without_future_positive_are_explicitly_excluded() -> None:
    result = evaluate_score_matrix(
        np.ones((2, 3)),
        np.asarray([[False, True, False], [False, False, False]]),
        np.zeros((2, 3), dtype=bool),
        [0, 1],
        k=1,
    )
    assert result["users_evaluated"] == 1
    assert result["users_without_future_positive"] == 1


def test_genre_exposure_and_preference_amplification_are_hard_topk() -> None:
    genres = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    recommendations = [np.asarray([0, 1]), np.asarray([2, 3])]
    histories = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=bool)
    result = genre_exposure_diagnostics(
        recommendations,
        [0, 1],
        genres,
        genre_names=["A", "B"],
        positive_histories=histories,
    )
    assert result["l1_between_genders"] == 2.0
    assert result["jensen_shannon_between_genders"] == 1.0
    assert result["preference_amplification"]["history_l1_between_genders"] == 2.0
    assert result["preference_amplification"]["scalar_l1_amplification"] == 0.0


def test_auc_is_tie_aware_and_balanced_accuracy_is_group_balanced() -> None:
    assert binary_auc(np.asarray([0, 1]), np.asarray([0.1, 0.9])) == 1.0
    assert binary_auc(np.asarray([0, 1]), np.asarray([0.5, 0.5])) == 0.5
    assert balanced_accuracy(
        np.asarray([0, 0, 0, 1]), np.asarray([0.1, 0.2, 0.9, 0.8])
    ) == pytest.approx((2 / 3 + 1.0) / 2)


def test_gender_probe_uses_separate_splits_and_finds_signal() -> None:
    rng = np.random.default_rng(7)

    def make_split(size: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.tile(np.asarray([0, 1]), size // 2)
        features = np.column_stack(
            (labels * 3.0 + rng.normal(0.0, 0.2, size), rng.normal(size=size))
        )
        return features, labels

    train_x, train_y = make_split(80)
    val_x, val_y = make_split(40)
    test_x, test_y = make_split(40)
    result = fit_gender_probe(
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        l2_values=(0.0, 0.01),
        steps=120,
    )
    assert result["test"]["auc"] > 0.95
    assert result["test"]["balanced_accuracy"] > 0.9
    assert result["protocol"]["report"] == "test users only"


def test_popularity_and_itemknn_only_use_fitted_positive_matrix() -> None:
    train = np.asarray([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    support = np.asarray([[0, 1, 0]], dtype=np.float32)
    popularity = PopularityRecommender().fit(train).score(support)
    assert popularity.tolist() == [[2.0, 1.0, 1.0]]
    scores = ItemKNNRecommender(neighbors=2, shrinkage=0.0).fit(train).score(support)
    assert scores.shape == (1, 3)
    assert scores[0, 0] > 0.0


def test_pareto_selection_respects_utility_tolerance() -> None:
    candidates = [
        {"ndcg@10": 0.40, "probe_auc_distance": 0.20},
        {"ndcg@10": 0.399, "probe_auc_distance": 0.05},
        {"ndcg@10": 0.30, "probe_auc_distance": 0.01},
        {"ndcg@10": 0.35, "probe_auc_distance": 0.30},
    ]
    assert pareto_front(candidates) == [0, 1, 2]
    assert select_pareto_candidate(candidates, utility_tolerance=0.01) == 1


def test_output_fair_selection_prioritizes_serving_gaps_inside_utility_floor() -> None:
    candidates = [
        {"ndcg@10": 1.0, "ndcg_gender_gap": 0.08, "genre_exposure_l1": 0.1, "exposure_residual_l1": 0.1, "probe_auc_distance": 0.01},
        {"ndcg@10": 0.995, "ndcg_gender_gap": 0.03, "genre_exposure_l1": 0.2, "exposure_residual_l1": 0.2, "probe_auc_distance": 0.2},
        {"ndcg@10": 0.98, "ndcg_gender_gap": 0.0, "genre_exposure_l1": 0.0, "exposure_residual_l1": 0.0, "probe_auc_distance": 0.0},
    ]
    assert select_output_fair_candidate(candidates, utility_tolerance=0.01) == 1


def test_deterministic_json_is_strict_and_sorted(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    write_json(path, {"z": np.float64(np.nan), "a": np.asarray([2, 1])})
    raw = path.read_text(encoding="utf-8")
    assert raw.index('"a"') < raw.index('"z"')
    assert json.loads(raw) == {"a": [2, 1], "z": None}

from __future__ import annotations

import numpy as np

from fairrec.evaluation import evaluate_score_matrix


def test_evaluation_masks_cold_logits_and_excludes_impossible_truth() -> None:
    result = evaluate_score_matrix(
        scores=np.asarray(
            [[3.0, 2.0, 100.0, 90.0], [3.0, 2.0, 100.0, 90.0]]
        ),
        query_positive=np.asarray(
            [[False, True, False, False], [False, False, True, False]]
        ),
        seen=np.asarray(
            [[True, False, False, False], [True, False, False, False]]
        ),
        genders=np.asarray([0, 1]),
        k=1,
        catalog_eligible=np.asarray([True, True, False, False]),
    )

    assert result["ndcg@1"] == 1.0
    assert result["users_evaluated"] == 1
    assert result["users_with_only_cold_future_positives"] == 1
    assert result["excluded_cold_query_positives"] == 1
    assert result["eligible_catalog_items"] == 2
    assert result["ineligible_catalog_items"] == 2
    assert result["coverage"] == 0.5


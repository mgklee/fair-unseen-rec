from __future__ import annotations

import numpy as np
import pytest
import torch

from fairrec.data import BalancedGenderBatchSampler, ML1MData, UserInteractionDataset


@pytest.fixture()
def tiny_data() -> ML1MData:
    # user0 ratings are chronological 5,1,4,3: the first two positive events
    # must be items 0 and 2 even though item1 is between them in wall-clock time.
    return ML1MData(
        movie_ids=np.asarray([10, 20, 30, 40, 50], dtype=np.int32),
        movie_titles=np.asarray(["a", "b", "c", "d", "e"]),
        movie_years=np.asarray([2000, 2001, 2002, 2003, 2004], dtype=np.int16),
        movie_genres=np.eye(5, dtype=np.uint8),
        user_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        user_genders=np.asarray([0, 0, 0, 1], dtype=np.uint8),
        user_splits=np.asarray([0, 0, 1, 2], dtype=np.uint8),
        user_ptr=np.asarray([0, 4, 6, 8, 10], dtype=np.int64),
        rating_item_indices=np.asarray([0, 1, 2, 3, 0, 4, 1, 3, 2, 4], dtype=np.int32),
        rating_values=np.asarray([5, 1, 4, 3, 4, 2, 5, 4, 4, 5], dtype=np.float32),
        rating_timestamps=np.asarray([1, 2, 3, 4, 1, 2, 1, 2, 1, 2], dtype=np.int64),
    )


def test_chronological_support_counts_positive_events(tiny_data: ML1MData) -> None:
    episode = tiny_data.chronological_episode(0, 1, positive_threshold=4)
    assert episode.positive_support_items.tolist() == [0]
    assert episode.positive_query_items.tolist() == [2]
    assert episode.support_ratings.tolist() == [5.0]


def test_dataset_is_binary_and_user_disjoint(tiny_data: ML1MData) -> None:
    dataset = UserInteractionDataset(tiny_data, split="train", positive_threshold=4)
    assert dataset.user_indices.tolist() == [0, 1]
    first = dataset[0]
    assert set(first) == {"input", "target", "gender", "user_index"}
    assert first["input"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert torch.equal(first["input"], first["target"])


def test_random_few_shot_view_is_deterministic_and_targets_full_profile(
    tiny_data: ML1MData,
) -> None:
    dataset = UserInteractionDataset(
        tiny_data,
        split="train",
        positive_threshold=4,
        random_support_sizes=(1,),
        full_profile_probability=0.0,
        seed=17,
    )
    dataset.set_epoch(3)
    first = dataset[0]
    repeated = dataset[0]
    assert first["input"].sum().item() == 1
    assert first["target"].sum().item() == 2
    assert torch.equal(first["input"], repeated["input"])
    assert first["input"].le(first["target"]).all()

    full = UserInteractionDataset(
        tiny_data,
        split="train",
        positive_threshold=4,
        random_support_sizes=(1,),
        full_profile_probability=1.0,
    )[0]
    assert torch.equal(full["input"], full["target"])


def test_balanced_sampler_oversamples_minority_and_covers_majority() -> None:
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    sampler = BalancedGenderBatchSampler(labels, batch_size=4, seed=9)
    batches = list(sampler)
    flattened = [position for batch in batches for position in batch]
    sampled_labels = labels[flattened]
    assert len(batches) == 3  # ceil(2 * majority_size / batch_size)
    assert np.count_nonzero(sampled_labels == 0) == np.count_nonzero(sampled_labels == 1)
    assert set(np.flatnonzero(labels == 0)).issubset(flattened)
    assert all(np.count_nonzero(labels[batch] == 0) == 2 for batch in batches)


def test_unsorted_history_is_rejected() -> None:
    with pytest.raises(ValueError, match="chronological"):
        ML1MData(
            movie_ids=np.asarray([1, 2], dtype=np.int32),
            movie_titles=np.asarray(["a", "b"]),
            movie_years=np.asarray([2000, 2001], dtype=np.int16),
            movie_genres=np.eye(2, dtype=np.uint8),
            user_ids=np.asarray([1], dtype=np.int32),
            user_genders=np.asarray([0], dtype=np.uint8),
            user_splits=np.asarray([0], dtype=np.uint8),
            user_ptr=np.asarray([0, 2], dtype=np.int64),
            rating_item_indices=np.asarray([0, 1], dtype=np.int32),
            rating_values=np.asarray([4, 5], dtype=np.float32),
            rating_timestamps=np.asarray([2, 1], dtype=np.int64),
        )

from __future__ import annotations

import numpy as np
import torch

from fairrec.checkpoint import save_checkpoint
from fairrec.config import ModelConfig
from fairrec.data import artifact_fingerprint, load_ml1m_npz
from fairrec.inference import MovieRecommender
from fairrec.model import MultVAE


def _write_tiny_artifact(path) -> None:
    np.savez_compressed(
        path,
        movie_ids=np.asarray([10, 20, 30, 40], dtype=np.int32),
        movie_titles=np.asarray(["warm a", "warm b", "cold c", "cold d"]),
        movie_years=np.asarray([2000, 2001, 2002, 2003], dtype=np.int16),
        movie_genres=np.eye(4, dtype=np.uint8),
        user_ids=np.asarray([1, 2], dtype=np.int32),
        user_genders=np.asarray([0, 1], dtype=np.uint8),
        user_splits=np.asarray([0, 2], dtype=np.uint8),
        user_ptr=np.asarray([0, 2, 4], dtype=np.int64),
        rating_item_indices=np.asarray([0, 1, 2, 3], dtype=np.int32),
        rating_values=np.asarray([5, 4, 5, 5], dtype=np.float32),
        rating_timestamps=np.asarray([1, 2, 1, 2], dtype=np.int64),
    )


def test_inference_excludes_coordinates_without_train_positive_signal(tmp_path) -> None:
    data_path = tmp_path / "tiny.npz"
    _write_tiny_artifact(data_path)
    data = load_ml1m_npz(data_path)
    assert data.warm_item_mask(positive_threshold=4).tolist() == [
        True,
        True,
        False,
        False,
    ]

    model = MultVAE(
        ModelConfig(num_items=4, hidden_dims=(), latent_dim=2, dropout=0.0)
    )
    with torch.no_grad():
        final = model.decoder[-1]
        assert isinstance(final, torch.nn.Linear)
        final.weight.zero_()
        # Give untrained coordinates overwhelming raw scores. Serving must
        # still return only the remaining warm item.
        final.bias.copy_(torch.tensor([0.0, 1.0, 100.0, 50.0]))
    checkpoint_path = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        epoch=1,
        global_step=1,
        experiment_config={"data": {"positive_threshold": 4.0}},
        data_fingerprint=artifact_fingerprint(data_path),
    )

    recommender = MovieRecommender(checkpoint_path, data_path)
    assert [movie.movie_id for movie in recommender.movies] == [10, 20]
    assert recommender.search("cold") == []
    rows = recommender.recommend([10], k=3)
    assert [row.movie_id for row in rows] == [20]


#!/usr/bin/env python3
"""Evaluate one MultVAE checkpoint on user-disjoint cold-start ML-1M users."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fairrec.checkpoint import load_model_checkpoint  # noqa: E402
from fairrec.data import GENRES, artifact_fingerprint, load_ml1m_npz  # noqa: E402
from fairrec.evaluation import (  # noqa: E402
    evaluate_score_matrix,
    fit_gender_probe,
    write_json,
)


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _stratified_subset(
    users: np.ndarray,
    genders: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    users = np.asarray(users, dtype=np.int64)
    if maximum is None or maximum >= users.size:
        return users
    if maximum < 2:
        raise ValueError("--max-users must be at least two")
    rng = np.random.default_rng(seed)
    male = users[genders[users] == 0]
    female = users[genders[users] == 1]
    male_count = min(maximum // 2, male.size)
    female_count = min(maximum - male_count, female.size)
    selected = np.concatenate(
        (
            rng.choice(male, male_count, replace=False),
            rng.choice(female, female_count, replace=False),
        )
    )
    if selected.size < maximum:
        unused = np.setdiff1d(users, selected)
        selected = np.concatenate(
            (selected, rng.choice(unused, maximum - selected.size, replace=False))
        )
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


def _episode_matrices(
    data: Any,
    users: Sequence[int] | np.ndarray,
    support_size: int,
    threshold: float,
    warm_catalog: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_array = np.asarray(users, dtype=np.int64)
    support = np.zeros((user_array.size, data.num_items), dtype=np.float32)
    query = np.zeros_like(support, dtype=bool)
    complete = np.zeros_like(support, dtype=bool)
    for row, user_index in enumerate(user_array):
        episode = data.chronological_episode(
            int(user_index),
            support_size,
            positive_threshold=threshold,
            item_eligible=warm_catalog,
        )
        support[row, episode.positive_support_items] = 1.0
        query[row, episode.positive_query_items] = True
        history = data.history(int(user_index))
        positive_items = history.item_indices[history.ratings >= threshold]
        if warm_catalog is not None:
            positive_items = positive_items[warm_catalog[positive_items]]
        complete[row, positive_items] = True
    return (
        support,
        query,
        data.user_genders[user_array].astype(np.int8, copy=False),
        complete,
    )


@torch.inference_mode()
def _model_arrays(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    score_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        output = model(batch, sample=False)
        score_parts.append(output.logits.float().cpu().numpy())
        latent_parts.append(output.mu.float().cpu().numpy())
    return np.concatenate(score_parts), np.concatenate(latent_parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data/processed/movielens_1m.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--support-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 20]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--probe-max-train-users", type=int)
    parser.add_argument("--probe-max-val-users", type=int)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.top_k <= 0 or args.batch_size <= 0:
        raise ValueError("top-k and batch-size must be positive")
    if any(size < 0 for size in args.support_sizes):
        raise ValueError("support sizes must be non-negative")

    device = _resolve_device(args.device)
    data = load_ml1m_npz(args.data)
    model, checkpoint = load_model_checkpoint(
        args.checkpoint, device=device, eval_mode=True
    )
    if model.config.num_items != data.num_items:
        raise ValueError("checkpoint and data catalog sizes differ")
    actual_fingerprint = artifact_fingerprint(args.data)
    expected_fingerprint = checkpoint.get("data_fingerprint")
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        raise ValueError("checkpoint and data fingerprints differ")

    evaluation_users = _stratified_subset(
        data.split_users(args.split), data.user_genders, args.max_users, args.seed
    )
    train_probe_users = _stratified_subset(
        data.split_users("train"),
        data.user_genders,
        args.probe_max_train_users,
        args.seed + 101,
    )
    val_probe_users = _stratified_subset(
        data.split_users("val"),
        data.user_genders,
        args.probe_max_val_users,
        args.seed + 211,
    )
    threshold = float(
        checkpoint.get("experiment_config", {})
        .get("data", {})
        .get("positive_threshold", 4.0)
    )
    warm_catalog = data.warm_item_mask(
        split="train", positive_threshold=threshold
    )
    output: dict[str, Any] = {
        "protocol": {
            "dataset": "MovieLens 1M",
            "split": args.split,
            "user_disjoint": True,
            "positive": f"rating >= {threshold:g}",
            "support": "first N timestamp-ordered positive interactions",
            "query": "subsequent positive interactions in stable timestamp order",
            "ranking": "train-warm catalog with positive support masked to -inf",
            "top_k": args.top_k,
            "support_sizes": args.support_sizes,
            "gender_use": "audit only; never supplied to recommender",
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model_type": checkpoint.get("model_type"),
        "data_fingerprint": actual_fingerprint,
        "split_users": int(evaluation_users.size),
        "support_results": {},
    }

    for support_size in args.support_sizes:
        support, query, genders, complete = _episode_matrices(
            data, evaluation_users, support_size, threshold, warm_catalog
        )
        scores, test_latents = _model_arrays(
            model, support, batch_size=args.batch_size, device=device
        )
        metrics = evaluate_score_matrix(
            scores,
            query,
            support.astype(bool),
            genders,
            k=args.top_k,
            movie_genres=data.movie_genres,
            genre_names=GENRES,
            positive_histories=support.astype(bool),
            catalog_eligible=warm_catalog,
        )
        if args.split == "test":
            train_support, _, train_gender, _ = _episode_matrices(
                data, train_probe_users, support_size, threshold, warm_catalog
            )
            val_support, _, val_gender, _ = _episode_matrices(
                data, val_probe_users, support_size, threshold, warm_catalog
            )
            _, train_latents = _model_arrays(
                model, train_support, batch_size=args.batch_size, device=device
            )
            _, val_latents = _model_arrays(
                model, val_support, batch_size=args.batch_size, device=device
            )
            metrics["posthoc_gender_probe"] = fit_gender_probe(
                train_latents,
                train_gender,
                val_latents,
                val_gender,
                test_latents,
                genders,
                seed=args.seed + support_size,
                steps=args.probe_steps,
            )
        else:
            metrics["posthoc_gender_probe"] = {
                "omitted": "test users are required for a final leakage-safe probe report"
            }
        output["support_results"][str(support_size)] = metrics

    write_json(args.output, output)
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

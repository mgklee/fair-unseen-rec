#!/usr/bin/env python3
"""Benchmark Popularity, ItemKNN, MultVAE, and Adv-MultVAE fairly on ML-1M."""

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
    ItemKNNRecommender,
    PopularityRecommender,
    evaluate_score_matrix,
    fit_gender_probe,
    write_json,
)


def _positive_matrix(data: Any, users: np.ndarray, threshold: float) -> np.ndarray:
    matrix = np.zeros((users.size, data.num_items), dtype=np.float32)
    for row, user_index in enumerate(users):
        history = data.history(int(user_index))
        matrix[row, history.item_indices[history.ratings >= threshold]] = 1.0
    return matrix


def _episode_matrices(
    data: Any,
    users: np.ndarray,
    support_size: int,
    threshold: float,
    warm_catalog: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = np.zeros((users.size, data.num_items), dtype=np.float32)
    query = np.zeros_like(support, dtype=bool)
    complete = np.zeros_like(support, dtype=bool)
    for row, user_index in enumerate(users):
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
    return support, query, data.user_genders[users].astype(np.int8), complete


def _stratified_subset(
    users: np.ndarray,
    genders: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is None or maximum >= users.size:
        return users
    if maximum < 2:
        raise ValueError("user limits must be at least two")
    rng = np.random.default_rng(seed)
    groups = [users[genders[users] == value] for value in (0, 1)]
    counts = [min(maximum // 2, groups[0].size), 0]
    counts[1] = min(maximum - counts[0], groups[1].size)
    selected = np.concatenate(
        [rng.choice(group, count, replace=False) for group, count in zip(groups, counts)]
    )
    if selected.size < maximum:
        remaining = np.setdiff1d(users, selected)
        selected = np.concatenate(
            (selected, rng.choice(remaining, maximum - selected.size, replace=False))
        )
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


@torch.inference_mode()
def _model_arrays(
    model: torch.nn.Module,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    for start in range(0, inputs.shape[0], batch_size):
        batch = torch.from_numpy(inputs[start : start + batch_size]).to(device)
        output = model(batch, sample=False)
        scores.append(output.logits.float().cpu().numpy())
        latents.append(output.mu.float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(latents)


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    return label.strip(), Path(raw_path)


def _evaluate(
    scores: np.ndarray,
    query: np.ndarray,
    support: np.ndarray,
    genders: np.ndarray,
    complete: np.ndarray,
    data: Any,
    warm_catalog: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    return evaluate_score_matrix(
        scores,
        query,
        support.astype(bool),
        genders,
        k=top_k,
        movie_genres=data.movie_genres,
        genre_names=GENRES,
        positive_histories=support.astype(bool),
        catalog_eligible=warm_catalog,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data/processed/movielens_1m.npz"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        default=[],
        metavar="LABEL=PATH",
        help="repeat for MultVAE and Adv-MultVAE checkpoints",
    )
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--adversarial-checkpoint", type=Path)
    parser.add_argument(
        "--support-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 20]
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--itemknn-neighbors", type=int, default=200)
    parser.add_argument("--itemknn-shrinkage", type=float, default=10.0)
    parser.add_argument("--max-test-users", type=int)
    parser.add_argument("--probe-max-train-users", type=int)
    parser.add_argument("--probe-max-val-users", type=int)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(size < 0 for size in args.support_sizes):
        raise ValueError("support sizes must be non-negative")
    checkpoints: list[tuple[str, Path]] = list(args.checkpoint)
    if args.baseline_checkpoint:
        checkpoints.append(("MultVAE", args.baseline_checkpoint))
    if args.adversarial_checkpoint:
        checkpoints.append(("Adv-MultVAE", args.adversarial_checkpoint))
    labels = [label for label, _ in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data = load_ml1m_npz(args.data)
    fingerprint = artifact_fingerprint(args.data)
    train_users = data.split_users("train")
    val_users = data.split_users("val")
    test_users = _stratified_subset(
        data.split_users("test"), data.user_genders, args.max_test_users, args.seed
    )
    probe_train_users = _stratified_subset(
        train_users, data.user_genders, args.probe_max_train_users, args.seed + 101
    )
    probe_val_users = _stratified_subset(
        val_users, data.user_genders, args.probe_max_val_users, args.seed + 211
    )
    threshold = 4.0
    train_positive = _positive_matrix(data, train_users, threshold)
    warm_catalog = train_positive.any(axis=0)
    popularity = PopularityRecommender().fit(train_positive)
    itemknn = ItemKNNRecommender(
        neighbors=args.itemknn_neighbors, shrinkage=args.itemknn_shrinkage
    ).fit(train_positive)

    loaded: list[tuple[str, torch.nn.Module, dict[str, Any], Path]] = []
    for label, path in checkpoints:
        model, payload = load_model_checkpoint(path, device=device, eval_mode=True)
        if model.config.num_items != data.num_items:
            raise ValueError(f"{label}: checkpoint and catalog sizes differ")
        expected = payload.get("data_fingerprint")
        if expected and expected != fingerprint:
            raise ValueError(f"{label}: checkpoint and data fingerprints differ")
        loaded.append((label, model, payload, path))

    deployment: dict[str, Any] = {}
    for label, _model, payload, path in loaded:
        model_type = str(payload.get("model_type"))
        role = (
            "naive"
            if model_type == "multvae"
            else "fair" if model_type == "adversarial_multvae" else label
        )
        deployment[role] = {
            "label": label,
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": artifact_fingerprint(path),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "model_type": model_type,
        }

    result: dict[str, Any] = {
        "protocol": {
            "dataset": "MovieLens 1M",
            "positive": "rating >= 4",
            "train": (
                "few-shot positive support inputs with complete-positive "
                "reconstruction targets"
            ),
            "evaluation": "user-disjoint test users",
            "support": "first N chronological positives",
            "query": "remaining later positives",
            "candidates": "all train-warm items; support and cold items set to -inf",
            "probe": "fit train latents, select validation, report test",
            "top_k": args.top_k,
        },
        "data_fingerprint": fingerprint,
        "deployment": deployment,
        "catalog_items": data.num_items,
        "warm_catalog_items": int(warm_catalog.sum()),
        "cold_catalog_items": int((~warm_catalog).sum()),
        "test_users": int(test_users.size),
        "models": {"Popularity": {}, "ItemKNN": {}},
    }

    for support_size in args.support_sizes:
        support, query, genders, complete = _episode_matrices(
            data, test_users, support_size, threshold, warm_catalog
        )
        result["models"]["Popularity"][str(support_size)] = _evaluate(
            popularity.score(support),
            query,
            support,
            genders,
            complete,
            data,
            warm_catalog,
            args.top_k,
        )
        result["models"]["ItemKNN"][str(support_size)] = _evaluate(
            itemknn.score(support),
            query,
            support,
            genders,
            complete,
            data,
            warm_catalog,
            args.top_k,
        )
        train_support, _, train_genders, _ = _episode_matrices(
            data, probe_train_users, support_size, threshold, warm_catalog
        )
        val_support, _, val_genders, _ = _episode_matrices(
            data, probe_val_users, support_size, threshold, warm_catalog
        )
        for label, model, payload, path in loaded:
            scores, test_latents = _model_arrays(
                model, support, args.batch_size, device
            )
            _, train_latents = _model_arrays(
                model, train_support, args.batch_size, device
            )
            _, val_latents = _model_arrays(model, val_support, args.batch_size, device)
            metrics = _evaluate(
                scores,
                query,
                support,
                genders,
                complete,
                data,
                warm_catalog,
                args.top_k,
            )
            metrics["posthoc_gender_probe"] = fit_gender_probe(
                train_latents,
                train_genders,
                val_latents,
                val_genders,
                test_latents,
                genders,
                seed=args.seed + support_size,
                steps=args.probe_steps,
            )
            model_results = result["models"].setdefault(
                label,
                {
                    "checkpoint": str(path.resolve()),
                    "model_type": payload.get("model_type"),
                    "checkpoint_epoch": payload.get("epoch"),
                    "support_results": {},
                },
            )
            model_results["support_results"][str(support_size)] = metrics

    write_json(args.output, result)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

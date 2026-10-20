#!/usr/bin/env python3
"""Train baseline/adversarial MultVAE with validation-only Pareto selection."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fairrec.checkpoint import save_checkpoint  # noqa: E402
from fairrec.config import ExperimentConfig  # noqa: E402
from fairrec.data import (  # noqa: E402
    BalancedGenderBatchSampler,
    UserInteractionDataset,
    artifact_fingerprint,
    load_ml1m_npz,
)
from fairrec.evaluation import (  # noqa: E402
    evaluate_score_matrix,
    fit_gender_probe,
    select_output_fair_candidate,
    write_json,
)
from fairrec.model import build_model  # noqa: E402
from fairrec.training import annealed_beta, train_epoch  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "non-JSON YAML requires PyYAML; the bundled .yaml files are "
                "JSON-compatible and need no optional dependency"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
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
    if maximum <= 1:
        raise ValueError("a stratified subset requires at least two users")
    rng = np.random.default_rng(seed)
    groups = [users[genders[users] == code] for code in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("both genders are required for stratified selection")
    first = min(maximum // 2, groups[0].size)
    second = min(maximum - first, groups[1].size)
    if first + second < maximum:
        remaining = np.setdiff1d(users, np.concatenate((groups[0][:first], groups[1][:second])))
        extra = rng.choice(remaining, maximum - first - second, replace=False)
    else:
        extra = np.empty(0, dtype=np.int64)
    selected = np.concatenate(
        (
            rng.choice(groups[0], first, replace=False),
            rng.choice(groups[1], second, replace=False),
            extra,
        )
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
    genders = data.user_genders[user_array].astype(np.int8, copy=False)
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
    return support, query, genders, complete


@torch.inference_mode()
def _model_arrays(
    model: torch.nn.Module,
    interactions: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    for start in range(0, interactions.shape[0], batch_size):
        batch = torch.from_numpy(interactions[start : start + batch_size]).to(device)
        output = model(batch, sample=False)
        scores.append(output.logits.float().cpu().numpy())
        # Deterministic posterior mean is the representation probed for gender.
        latents.append(output.mu.float().cpu().numpy())
    return np.concatenate(scores), np.concatenate(latents)


def _validation_metrics(
    model: torch.nn.Module,
    data: Any,
    *,
    train_users: np.ndarray,
    val_users: np.ndarray,
    support_size: int,
    threshold: float,
    top_k: int,
    batch_size: int,
    device: torch.device,
    probe_steps: int,
    probe_l2_values: Sequence[float],
    seed: int,
    warm_catalog: np.ndarray,
) -> dict[str, Any]:
    train_support, _, train_gender, _ = _episode_matrices(
        data, train_users, support_size, threshold, warm_catalog
    )
    val_support, val_query, val_gender, val_complete = _episode_matrices(
        data, val_users, support_size, threshold, warm_catalog
    )
    _, train_latent = _model_arrays(
        model, train_support, batch_size=batch_size, device=device
    )
    val_scores, val_latent = _model_arrays(
        model, val_support, batch_size=batch_size, device=device
    )
    ranking = evaluate_score_matrix(
        val_scores,
        val_query,
        val_support.astype(bool),
        val_gender,
        k=top_k,
        movie_genres=data.movie_genres,
        positive_histories=val_support.astype(bool),
        catalog_eligible=warm_catalog,
    )
    # The third split is intentionally the validation split again: its returned
    # "test" value is discarded. This reuses one rigorously split probe helper
    # while ensuring checkpoint selection never reads true test users.
    probe = fit_gender_probe(
        train_latent,
        train_gender,
        val_latent,
        val_gender,
        val_latent,
        val_gender,
        l2_values=probe_l2_values,
        seed=seed,
        steps=probe_steps,
    )
    return {
        "ranking": ranking,
        "probe_validation": probe["validation"],
        "probe_selected_l2": probe["selected_l2"],
        "probe_selected_threshold": probe["selected_threshold"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-type",
        choices=("baseline", "adversarial", "multvae", "adversarial_multvae"),
        help="override the YAML model_type (baseline=multvae)",
    )
    parser.add_argument("--device", help="override train.device, e.g. cpu or cuda:0")
    parser.add_argument("--epochs", type=int, help="bounded override for smoke runs")
    parser.add_argument("--grl-lambda", type=float, help="override train.grl_lambda")
    parser.add_argument("--seed", type=int, help="override train.seed")
    parser.add_argument(
        "--genre-fairness-weight", type=float,
        help="override train.genre_fairness_weight",
    )
    parser.add_argument(
        "--genre-fairness-mode",
        choices=("parity", "support_residual"),
        help="override train.genre_fairness_mode",
    )
    parser.add_argument(
        "--genre-temperature", type=float,
        help="override train.genre_temperature",
    )
    args = parser.parse_args()

    raw_config = _load_yaml(args.config)
    experiment = ExperimentConfig.from_dict(raw_config)
    model_aliases = {"baseline": "multvae", "adversarial": "adversarial_multvae"}
    if args.model_type:
        experiment = replace(
            experiment,
            model_type=model_aliases.get(args.model_type, args.model_type),
        )
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        experiment = replace(experiment, train=replace(experiment.train, epochs=args.epochs))
    if args.grl_lambda is not None:
        if args.grl_lambda < 0:
            raise ValueError("--grl-lambda must be non-negative")
        experiment = replace(
            experiment, train=replace(experiment.train, grl_lambda=args.grl_lambda)
        )
    if args.seed is not None:
        experiment = replace(experiment, train=replace(experiment.train, seed=args.seed))
    if args.genre_fairness_weight is not None:
        if args.genre_fairness_weight < 0:
            raise ValueError("--genre-fairness-weight must be non-negative")
        experiment = replace(
            experiment,
            train=replace(
                experiment.train,
                genre_fairness_weight=args.genre_fairness_weight,
            ),
        )
    if args.genre_fairness_mode is not None:
        experiment = replace(
            experiment,
            train=replace(experiment.train, genre_fairness_mode=args.genre_fairness_mode),
        )
    if args.genre_temperature is not None:
        if args.genre_temperature <= 0:
            raise ValueError("--genre-temperature must be positive")
        experiment = replace(
            experiment,
            train=replace(experiment.train, genre_temperature=args.genre_temperature),
        )
    device = _resolve_device(args.device or experiment.train.device)
    _set_determinism(experiment.train.seed)

    data_path = Path(experiment.data.npz_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    data = load_ml1m_npz(data_path)
    if experiment.model.num_items != data.num_items:
        raise ValueError(
            f"config has {experiment.model.num_items} items but artifact has {data.num_items}"
        )
    selection = dict(raw_config.get("selection", {}))
    train_users = _stratified_subset(
        data.split_users("train"),
        data.user_genders,
        selection.get("train_users"),
        experiment.train.seed,
    )
    probe_train_users = _stratified_subset(
        train_users,
        data.user_genders,
        selection.get("probe_train_users"),
        experiment.train.seed + 11,
    )
    val_users = _stratified_subset(
        data.split_users("val"),
        data.user_genders,
        selection.get("probe_validation_users"),
        experiment.train.seed + 17,
    )

    train_dataset = UserInteractionDataset(
        data,
        user_indices=train_users,
        positive_threshold=experiment.data.positive_threshold,
        random_support_sizes=selection.get("training_support_sizes"),
        full_profile_probability=float(selection.get("full_profile_probability", 0.0)),
        seed=experiment.train.seed,
    )
    generator = torch.Generator().manual_seed(experiment.train.seed)
    # Both arms use the same natural-gender minibatch distribution so that the
    # only intervention is the adversarial regularizer.
    sampler = None
    loader = DataLoader(
        train_dataset,
        batch_size=experiment.train.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = build_model(experiment.model, experiment.model_type).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=experiment.train.learning_rate,
        weight_decay=experiment.train.weight_decay,
    )

    output_dir = args.output_dir.resolve()
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = artifact_fingerprint(data_path)
    candidates: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    global_step = 0
    support_size = int(selection.get("support_size", 5))
    top_k = int(selection.get("top_k", 10))
    warm_catalog = data.warm_item_mask(
        split="train", positive_threshold=experiment.data.positive_threshold
    )
    genre_features = torch.as_tensor(
        data.movie_genres, dtype=torch.float32, device=device
    )
    warm_catalog_tensor = torch.as_tensor(
        warm_catalog, dtype=torch.bool, device=device
    )
    probe_steps = int(selection.get("probe_steps", 200))
    probe_l2_values = tuple(
        float(value) for value in selection.get("probe_l2_values", (1e-4, 1e-3, 1e-2))
    )

    for epoch in range(1, experiment.train.epochs + 1):
        train_dataset.set_epoch(epoch - 1)
        if sampler is not None:
            sampler.set_epoch(epoch - 1)
        trained = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            global_step=global_step,
            max_beta=experiment.train.max_beta,
            kl_anneal_steps=experiment.train.kl_anneal_steps,
            adversary_weight=experiment.train.adversary_weight,
            grl_lambda=experiment.train.grl_lambda,
            genre_fairness_weight=experiment.train.genre_fairness_weight,
            genre_temperature=experiment.train.genre_temperature,
            genre_fairness_mode=experiment.train.genre_fairness_mode,
            movie_genres=genre_features,
            catalog_eligible=warm_catalog_tensor,
            grad_clip=experiment.train.grad_clip,
        )
        global_step = trained.global_step
        validation = _validation_metrics(
            model,
            data,
            train_users=probe_train_users,
            val_users=val_users,
            support_size=support_size,
            threshold=experiment.data.positive_threshold,
            top_k=top_k,
            batch_size=experiment.train.batch_size,
            device=device,
            probe_steps=probe_steps,
            probe_l2_values=probe_l2_values,
            seed=experiment.train.seed,
            warm_catalog=warm_catalog,
        )
        utility = float(validation["ranking"][f"ndcg@{top_k}"])
        validation_auc = float(validation["probe_validation"]["auc"])
        utility_gap = float(
            validation["ranking"]["absolute_gender_gaps"][f"ndcg@{top_k}"]
        )
        exposure = validation["ranking"]["genre_exposure"]
        exposure_residual = float(
            exposure["preference_amplification"]["residual_l1"]
        )
        candidate = {
            "epoch": epoch,
            f"ndcg@{top_k}": utility,
            "probe_auc": validation_auc,
            "probe_auc_distance": abs(validation_auc - 0.5),
            "ndcg_gender_gap": utility_gap,
            "genre_exposure_l1": float(exposure["l1_between_genders"]),
            "exposure_residual_l1": exposure_residual,
            "training": {
                "loss": trained.loss,
                "reconstruction": trained.reconstruction,
                "kl": trained.kl,
                "adversary_loss": trained.adversary_loss,
                "adversary_accuracy": trained.adversary_accuracy,
                "genre_fairness_loss": trained.genre_fairness_loss,
                "global_step": global_step,
                "beta_at_epoch_end": annealed_beta(
                    global_step,
                    experiment.train.max_beta,
                    experiment.train.kl_anneal_steps,
                ),
            },
            "validation": validation,
        }
        candidates.append(candidate)
        checkpoint_path = candidate_dir / f"epoch_{epoch:04d}.pt"
        checkpoint_paths.append(checkpoint_path)
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=epoch,
            global_step=global_step,
            metrics=candidate,
            experiment_config={
                **experiment.to_dict(),
                "selection": selection,
                "effective_device": str(device),
                "cli_model_type_override": args.model_type,
                "cli_epochs_override": args.epochs,
                "cli_grl_lambda_override": args.grl_lambda,
                "cli_seed_override": args.seed,
                "cli_genre_fairness_weight_override": args.genre_fairness_weight,
                "cli_genre_fairness_mode_override": args.genre_fairness_mode,
                "cli_genre_temperature_override": args.genre_temperature,
            },
            data_fingerprint=fingerprint,
        )
        print(json.dumps({"epoch": epoch, "ndcg": utility, "probe_auc": validation_auc}))

    if experiment.model_type == "multvae":
        # A genuinely naive control: validation utility only, never fairness.
        selected_index = max(
            range(len(candidates)),
            key=lambda index: (candidates[index][f"ndcg@{top_k}"], -index),
        )
        selection_rule = "validation NDCG only (naive control)"
    else:
        selected_index = select_output_fair_candidate(
            candidates,
            utility_key=f"ndcg@{top_k}",
            utility_tolerance=float(selection.get("utility_tolerance", 0.01)),
        )
        selection_rule = (
            "validation output fairness: NDCG floor, then gender NDCG gap, "
            "hard genre exposure gap, support-conditioned residual, and probe leakage"
        )
    best_path = output_dir / "best.pt"
    shutil.copy2(checkpoint_paths[selected_index], best_path)
    keep_candidates = bool(selection.get("keep_candidate_checkpoints", False))
    if not keep_candidates:
        shutil.rmtree(candidate_dir)
    summary = {
        "protocol": {
            "training_input": (
                "deterministic per-epoch random few-shot positives; complete profile target"
                if train_dataset.random_support_sizes is not None
                else "complete positive vectors from train users only"
            ),
            "training_support_sizes": train_dataset.random_support_sizes,
            "full_profile_probability": train_dataset.full_profile_probability,
            "positive_threshold": experiment.data.positive_threshold,
            "checkpoint_selection": selection_rule,
            "minibatch_gender_distribution": "natural and identical across arms",
            "utility": f"NDCG@{top_k}",
            "fairness_regularization": {
                "grl_lambda": experiment.train.grl_lambda,
                "adversary_weight": experiment.train.adversary_weight,
                "soft_gbs_weight": experiment.train.genre_fairness_weight,
                "soft_gbs_temperature": experiment.train.genre_temperature,
                "soft_gbs_mode": experiment.train.genre_fairness_mode,
            },
            "checkpoint_fairness_audit": (
                "not used for naive control"
                if experiment.model_type == "multvae"
                else "validation gender NDCG gap, hard genre exposure L1, "
                "support-conditioned residual, then probe leakage"
            ),
            "representation_audit": (
                "direction-invariant distance of validation gender-probe AUC from 0.5"
            ),
            "kl_anneal_steps": experiment.train.kl_anneal_steps,
            "final_effective_beta": annealed_beta(
                global_step,
                experiment.train.max_beta,
                experiment.train.kl_anneal_steps,
            ),
            "test_users_touched": False,
            "candidate_catalog": "items liked by at least one train user",
            "warm_catalog_items": int(warm_catalog.sum()),
            "candidate_checkpoints_retained": keep_candidates,
        },
        "model_type": experiment.model_type,
        "data_fingerprint": fingerprint,
        "selected_candidate_index": selected_index,
        "selected_epoch": candidates[selected_index]["epoch"],
        "selected_checkpoint": str(best_path),
        "candidates": candidates,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps({"selected_epoch": summary["selected_epoch"], "checkpoint": str(best_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

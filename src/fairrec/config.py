"""Typed experiment configuration with dependency-free JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class DataConfig:
    """MovieLens artifact and genuinely user-disjoint evaluation settings."""

    npz_path: str = "data/processed/movielens_1m.npz"
    positive_threshold: float = 4.0
    support_sizes: tuple[int, ...] = (1, 3, 5, 10, 20)

    def __post_init__(self) -> None:
        if not 1.0 <= self.positive_threshold <= 5.0:
            raise ValueError("positive_threshold must be in [1, 5]")
        if not self.support_sizes or any(size < 0 for size in self.support_sizes):
            raise ValueError("support_sizes must contain non-negative event counts")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Dimensions shared by the baseline and adversarial MultVAE models."""

    num_items: int
    hidden_dims: tuple[int, ...] = (600,)
    latent_dim: int = 200
    dropout: float = 0.5
    adversary_hidden_dims: tuple[int, ...] = (128, 64)
    num_gender_classes: int = 2

    def __post_init__(self) -> None:
        if self.num_items <= 1:
            raise ValueError("num_items must be greater than one")
        if self.latent_dim <= 0 or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("all model dimensions must be positive")
        if any(dim <= 0 for dim in self.adversary_hidden_dims):
            raise ValueError("all adversary dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.num_gender_classes < 2:
            raise ValueError("num_gender_classes must be at least two")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelConfig":
        payload = dict(value)
        for key in ("hidden_dims", "adversary_hidden_dims"):
            if key in payload:
                payload[key] = tuple(int(part) for part in payload[key])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimization settings for simultaneous GRL adversarial training."""

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_beta: float = 0.2
    kl_anneal_steps: int = 200_000
    adversary_weight: float = 1.0
    grl_lambda: float = 1.0
    genre_fairness_weight: float = 0.0
    genre_temperature: float = 1.0
    genre_fairness_mode: str = "parity"
    grad_clip: float = 5.0
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.max_beta < 0 or self.kl_anneal_steps < 0:
            raise ValueError("KL settings must be non-negative")
        if (
            self.adversary_weight < 0
            or self.grl_lambda < 0
            or self.genre_fairness_weight < 0
        ):
            raise ValueError("fairness coefficients must be non-negative")
        if self.genre_temperature <= 0:
            raise ValueError("genre_temperature must be positive")
        if self.genre_fairness_mode not in {"parity", "support_residual"}:
            raise ValueError("genre_fairness_mode must be parity or support_residual")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete reproducible experiment configuration."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    model_type: str = "adversarial_multvae"

    def __post_init__(self) -> None:
        if self.model_type not in {"multvae", "adversarial_multvae"}:
            raise ValueError("model_type must be multvae or adversarial_multvae")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        payload = dict(value)
        return cls(
            data=DataConfig(**dict(payload["data"])),
            model=ModelConfig.from_dict(payload["model"]),
            train=TrainConfig(**dict(payload.get("train", {}))),
            model_type=str(payload.get("model_type", "adversarial_multvae")),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


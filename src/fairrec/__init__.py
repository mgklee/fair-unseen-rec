"""Public API for fair, inductive MovieLens recommendation."""

from .checkpoint import load_checkpoint, load_model_checkpoint, save_checkpoint
from .config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from .data import (
    BalancedGenderBatchSampler,
    ChronologicalEpisode,
    ML1MData,
    Movie,
    UserInteractionDataset,
    artifact_fingerprint,
    load_ml1m_npz,
)
from .inference import MovieRecommender, Recommendation
from .model import (
    AdversarialMultVAE,
    AdversarialVAEOutput,
    MultVAE,
    VAEOutput,
    build_model,
    gradient_reverse,
)
from .training import (
    LossBreakdown,
    TrainEpochResult,
    adversarial_loss,
    annealed_beta,
    kl_divergence,
    multinomial_reconstruction_loss,
    multvae_loss,
    soft_genre_residual_loss,
    train_epoch,
)

__all__ = [
    "AdversarialMultVAE",
    "AdversarialVAEOutput",
    "BalancedGenderBatchSampler",
    "ChronologicalEpisode",
    "DataConfig",
    "ExperimentConfig",
    "LossBreakdown",
    "ML1MData",
    "ModelConfig",
    "Movie",
    "MovieRecommender",
    "MultVAE",
    "Recommendation",
    "TrainConfig",
    "TrainEpochResult",
    "UserInteractionDataset",
    "VAEOutput",
    "adversarial_loss",
    "annealed_beta",
    "artifact_fingerprint",
    "build_model",
    "gradient_reverse",
    "kl_divergence",
    "load_checkpoint",
    "load_ml1m_npz",
    "load_model_checkpoint",
    "multinomial_reconstruction_loss",
    "multvae_loss",
    "soft_genre_residual_loss",
    "save_checkpoint",
    "train_epoch",
]

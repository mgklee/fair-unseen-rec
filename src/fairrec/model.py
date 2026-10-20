"""Inductive MultVAE models with a gradient-reversal gender adversary."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig


class VAEOutput(NamedTuple):
    logits: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


class AdversarialVAEOutput(NamedTuple):
    logits: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor
    gender_logits: Tensor


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = float(coefficient)  # type: ignore[attr-defined]
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, None]:
        coefficient = ctx.coefficient  # type: ignore[attr-defined]
        return gradient.neg().mul(coefficient), None


def gradient_reverse(value: Tensor, coefficient: float = 1.0) -> Tensor:
    """Identity forward pass whose backward gradient is ``-coefficient * grad``."""

    if coefficient < 0:
        raise ValueError("gradient-reversal coefficient must be non-negative")
    return _GradientReversal.apply(value, float(coefficient))


def _tanh_mlp(dimensions: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        layers.extend((nn.Linear(input_dim, output_dim), nn.Tanh()))
    return nn.Sequential(*layers)


class MultVAE(nn.Module):
    """User-ID-free multinomial VAE over a user's binary positive-item vector.

    The shared catalog defines the input/output coordinates.  Consequently a
    new user can be encoded immediately from any non-empty set of liked movies;
    no user embedding or demographic field exists in this interface.
    """

    model_type = "multvae"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        encoder_dimensions = [config.num_items, *config.hidden_dims]
        encoder_output_dim = encoder_dimensions[-1]
        self.input_dropout = nn.Dropout(config.dropout)
        self.encoder = _tanh_mlp(encoder_dimensions)
        self.mu_layer = nn.Linear(encoder_output_dim, config.latent_dim)
        self.logvar_layer = nn.Linear(encoder_output_dim, config.latent_dim)

        decoder_dimensions = [
            config.latent_dim,
            *reversed(config.hidden_dims),
            config.num_items,
        ]
        decoder_layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(decoder_dimensions[:-1], decoder_dimensions[1:])
        ):
            decoder_layers.append(nn.Linear(input_dim, output_dim))
            if index < len(decoder_dimensions) - 2:
                decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_input(self, interactions: Tensor) -> Tensor:
        if interactions.ndim != 2 or interactions.shape[1] != self.config.num_items:
            raise ValueError(
                f"interactions must be [batch, {self.config.num_items}]"
            )
        if not interactions.is_floating_point():
            interactions = interactions.float()
        if torch.any(interactions < 0):
            raise ValueError("interactions must be non-negative")
        return interactions

    def encode(self, interactions: Tensor) -> tuple[Tensor, Tensor]:
        """Encode only observed positive interactions; no identity/demographics."""

        interactions = self._validate_input(interactions)
        corrupted = self.input_dropout(interactions)
        normalized = F.normalize(corrupted, p=2, dim=1, eps=1e-8)
        hidden = self.encoder(normalized)
        mu = self.mu_layer(hidden)
        # Bounding variance avoids exp overflow on sparse/empty prefix vectors.
        logvar = self.logvar_layer(hidden).clamp(min=-10.0, max=10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, *, sample: bool) -> Tensor:
        if not sample:
            return mu
        standard_deviation = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(standard_deviation) * standard_deviation

    def decode(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dim:
            raise ValueError(f"latent must be [batch, {self.config.latent_dim}]")
        return self.decoder(latent)

    def forward(
        self,
        interactions: Tensor,
        *,
        sample: bool | None = None,
    ) -> VAEOutput:
        mu, logvar = self.encode(interactions)
        z = self.reparameterize(
            mu, logvar, sample=self.training if sample is None else bool(sample)
        )
        return VAEOutput(self.decode(z), mu, logvar, z)

    def score(self, interactions: Tensor) -> Tensor:
        """Deterministic full-catalog logits used for validation and inference."""

        mu, _ = self.encode(interactions)
        return self.decode(mu)


class GenderAdversary(nn.Module):
    """Training-only classifier attempting to recover gender from VAE means."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimensions = [
            config.latent_dim,
            *config.adversary_hidden_dims,
            config.num_gender_classes,
        ]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.extend((nn.LeakyReLU(0.2), nn.Dropout(config.dropout / 2.0)))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: Tensor) -> Tensor:
        return self.network(latent)


class AdversarialMultVAE(MultVAE):
    """MultVAE whose encoder is trained to make gender unrecoverable via GRL."""

    model_type = "adversarial_multvae"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.gender_adversary = GenderAdversary(config)
        # The subclass module is created after MultVAE's reset pass.
        for module in self.gender_adversary.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        interactions: Tensor,
        *,
        grl_lambda: float = 1.0,
        sample: bool | None = None,
    ) -> AdversarialVAEOutput:
        base = super().forward(interactions, sample=sample)
        gender_logits = self.gender_adversary(
            gradient_reverse(base.mu, coefficient=grl_lambda)
        )
        return AdversarialVAEOutput(*base, gender_logits)


def build_model(
    config: ModelConfig,
    model_type: str = "adversarial_multvae",
) -> MultVAE:
    if model_type == "multvae":
        return MultVAE(config)
    if model_type == "adversarial_multvae":
        return AdversarialMultVAE(config)
    raise ValueError(f"unknown model_type: {model_type!r}")


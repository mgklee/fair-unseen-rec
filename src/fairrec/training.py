"""Losses and bounded training primitives for baseline/adversarial MultVAE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import AdversarialMultVAE, AdversarialVAEOutput, MultVAE, VAEOutput


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    adversary: Tensor


@dataclass(frozen=True, slots=True)
class TrainEpochResult:
    loss: float
    reconstruction: float
    kl: float
    adversary_loss: float
    adversary_accuracy: float
    genre_fairness_loss: float
    examples: int
    steps: int
    global_step: int


def multinomial_reconstruction_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Mean negative multinomial log-likelihood over positive catalog entries."""

    if logits.shape != target.shape or logits.ndim != 2:
        raise ValueError("logits and target must have the same [batch, items] shape")
    per_user = -(F.log_softmax(logits, dim=1) * target.float()).sum(dim=1)
    return per_user.mean()


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    if mu.shape != logvar.shape or mu.ndim != 2:
        raise ValueError("mu and logvar must have the same [batch, latent] shape")
    return (-0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean()


def multvae_loss(
    output: VAEOutput,
    target: Tensor,
    *,
    beta: float,
) -> LossBreakdown:
    if beta < 0:
        raise ValueError("beta must be non-negative")
    reconstruction = multinomial_reconstruction_loss(output.logits, target)
    kl = kl_divergence(output.mu, output.logvar)
    zero = reconstruction.new_zeros(())
    return LossBreakdown(reconstruction + float(beta) * kl, reconstruction, kl, zero)


def adversarial_loss(
    output: AdversarialVAEOutput,
    target: Tensor,
    genders: Tensor,
    *,
    beta: float,
    adversary_weight: float,
) -> LossBreakdown:
    """Joint objective; GRL supplies the negative sign to encoder gradients."""

    if adversary_weight < 0:
        raise ValueError("adversary_weight must be non-negative")
    if genders.ndim != 1 or genders.shape[0] != target.shape[0]:
        raise ValueError("genders must have one class label per user")
    if torch.any((genders < 0) | (genders >= output.gender_logits.shape[1])):
        raise ValueError("gender contains a class unsupported by the adversary")
    reconstruction = multinomial_reconstruction_loss(output.logits, target)
    kl = kl_divergence(output.mu, output.logvar)
    classifier = F.cross_entropy(output.gender_logits, genders.long())
    total = reconstruction + float(beta) * kl + float(adversary_weight) * classifier
    return LossBreakdown(total, reconstruction, kl, classifier)


def annealed_beta(global_step: int, max_beta: float, anneal_steps: int) -> float:
    if max_beta < 0 or anneal_steps < 0:
        raise ValueError("KL annealing values must be non-negative")
    if anneal_steps == 0:
        return float(max_beta)
    return float(max_beta) * min(1.0, max(0, int(global_step)) / anneal_steps)


def soft_genre_residual_loss(
    logits: Tensor,
    inputs: Tensor,
    genders: Tensor,
    movie_genres: Tensor,
    *,
    temperature: float = 1.0,
    catalog_eligible: Tensor | None = None,
    mode: str = "support_residual",
) -> Tensor:
    """Differentiable genre-exposure parity or support-residual surrogate.

    Softmax scores approximate recommendation exposure. ``parity`` minimizes
    the M/F predicted genre gap (the soft GBS-style objective); the alternative
    matches that gap to observed support preferences. Evaluation remains hard
    Top-K and reports preference amplification to detect taste erasure.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode not in {"parity", "support_residual"}:
        raise ValueError("mode must be parity or support_residual")
    if logits.shape != inputs.shape or logits.ndim != 2:
        raise ValueError("logits and inputs must share [batch, items] shape")
    if genders.ndim != 1 or genders.shape[0] != logits.shape[0]:
        raise ValueError("genders must contain one label per row")
    if movie_genres.ndim != 2 or movie_genres.shape[0] != logits.shape[1]:
        raise ValueError("movie_genres must be [items, genres]")
    adjusted = logits / float(temperature)
    invalid = inputs > 0
    if catalog_eligible is not None:
        eligible = catalog_eligible.bool().reshape(-1)
        if eligible.shape != (logits.shape[1],):
            raise ValueError("catalog_eligible must contain one value per item")
        invalid = invalid | ~eligible.unsqueeze(0)
    adjusted = adjusted.masked_fill(invalid, -torch.inf)
    probabilities = torch.softmax(adjusted, dim=1)
    features = movie_genres.to(device=logits.device, dtype=logits.dtype)
    predicted = probabilities @ features
    predicted = predicted / predicted.sum(dim=1, keepdim=True).clamp_min(1e-8)
    observed = inputs @ features
    observed = observed / observed.sum(dim=1, keepdim=True).clamp_min(1e-8)
    valid_history = inputs.sum(dim=1) > 0
    group_means: list[tuple[Tensor, Tensor]] = []
    for code in (0, 1):
        selected = (genders == code) & valid_history
        if not bool(selected.any()):
            return logits.new_zeros(())
        group_means.append(
            (predicted[selected].mean(dim=0), observed[selected].mean(dim=0))
        )
    predicted_gap = group_means[1][0] - group_means[0][0]
    if mode == "parity":
        return torch.abs(predicted_gap).sum()
    observed_gap = (group_means[1][1] - group_means[0][1]).detach()
    return torch.abs(predicted_gap - observed_gap).sum()


def _batch_value(batch: Mapping[str, Tensor], key: str, device: torch.device) -> Tensor:
    try:
        value = batch[key]
    except KeyError as exc:
        raise KeyError(f"training batch is missing {key!r}") from exc
    return value.to(device, non_blocking=True)


def train_epoch(
    model: MultVAE,
    loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    global_step: int = 0,
    max_beta: float = 0.2,
    kl_anneal_steps: int = 200_000,
    adversary_weight: float = 1.0,
    grl_lambda: float = 1.0,
    genre_fairness_weight: float = 0.0,
    genre_temperature: float = 1.0,
    genre_fairness_mode: str = "parity",
    movie_genres: Tensor | None = None,
    catalog_eligible: Tensor | None = None,
    grad_clip: float = 5.0,
) -> TrainEpochResult:
    """Train one epoch. Input dropout is owned by the model and applied once."""

    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    target_device = torch.device(device)
    model.train()
    aggregate = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "adversary": 0.0,
        "genre_fairness": 0.0,
    }
    correct = 0
    examples = 0
    steps = 0

    for batch in loader:
        inputs = _batch_value(batch, "input", target_device).float()
        targets = _batch_value(batch, "target", target_device).float()
        batch_size = int(inputs.shape[0])
        if batch_size == 0:
            continue
        beta = annealed_beta(global_step, max_beta, kl_anneal_steps)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, AdversarialMultVAE):
            genders = _batch_value(batch, "gender", target_device).long()
            output = model(inputs, grl_lambda=grl_lambda, sample=True)
            losses = adversarial_loss(
                output,
                targets,
                genders,
                beta=beta,
                adversary_weight=adversary_weight,
            )
            genre_penalty = inputs.new_zeros(())
            if genre_fairness_weight > 0:
                if movie_genres is None:
                    raise ValueError("movie_genres is required for genre fairness")
                genre_penalty = soft_genre_residual_loss(
                    output.logits,
                    inputs,
                    genders,
                    movie_genres,
                    temperature=genre_temperature,
                    catalog_eligible=catalog_eligible,
                    mode=genre_fairness_mode,
                )
            correct += int((output.gender_logits.argmax(dim=1) == genders).sum().item())
        else:
            output = model(inputs, sample=True)
            losses = multvae_loss(output, targets, beta=beta)
            genre_penalty = inputs.new_zeros(())

        total_loss = losses.total + float(genre_fairness_weight) * genre_penalty
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite MultVAE training loss")
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        aggregate["loss"] += float(total_loss.detach()) * batch_size
        aggregate["reconstruction"] += float(losses.reconstruction.detach()) * batch_size
        aggregate["kl"] += float(losses.kl.detach()) * batch_size
        aggregate["adversary"] += float(losses.adversary.detach()) * batch_size
        aggregate["genre_fairness"] += float(genre_penalty.detach()) * batch_size
        examples += batch_size
        steps += 1
        global_step += 1

    denominator = max(examples, 1)
    return TrainEpochResult(
        loss=aggregate["loss"] / denominator,
        reconstruction=aggregate["reconstruction"] / denominator,
        kl=aggregate["kl"] / denominator,
        adversary_loss=aggregate["adversary"] / denominator,
        adversary_accuracy=correct / denominator if isinstance(model, AdversarialMultVAE) else float("nan"),
        genre_fairness_loss=aggregate["genre_fairness"] / denominator,
        examples=examples,
        steps=steps,
        global_step=global_step,
    )


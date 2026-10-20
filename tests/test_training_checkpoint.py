from __future__ import annotations

import inspect

import torch

from fairrec.checkpoint import load_model_checkpoint, save_checkpoint
from fairrec.config import ModelConfig
from fairrec.inference import MovieRecommender
from fairrec.model import AdversarialMultVAE
from fairrec.training import adversarial_loss, soft_genre_residual_loss


def test_adversarial_loss_is_finite_and_backpropagates() -> None:
    model = AdversarialMultVAE(
        ModelConfig(num_items=9, hidden_dims=(7,), latent_dim=4, dropout=0.0)
    )
    inputs = torch.eye(9)[:4]
    output = model(inputs, sample=False)
    breakdown = adversarial_loss(
        output,
        inputs,
        torch.tensor([0, 1, 0, 1]),
        beta=0.1,
        adversary_weight=0.5,
    )
    breakdown.total.backward()
    assert torch.isfinite(breakdown.total)
    assert model.mu_layer.weight.grad is not None
    assert model.gender_adversary.network[0].weight.grad is not None


def test_soft_genre_residual_is_differentiable_and_support_conditioned() -> None:
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 3.0]],
        requires_grad=True,
    )
    inputs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    genres = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    penalty = soft_genre_residual_loss(
        logits,
        inputs,
        torch.tensor([0, 1]),
        genres,
        catalog_eligible=torch.ones(4, dtype=torch.bool),
    )
    penalty.backward()
    assert torch.isfinite(penalty)
    assert penalty.item() > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_checkpoint_round_trip(tmp_path) -> None:
    config = ModelConfig(num_items=9, hidden_dims=(7,), latent_dim=4, dropout=0.0)
    model = AdversarialMultVAE(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, epoch=3, global_step=17, metrics={"ndcg": 0.2})
    restored, payload = load_model_checkpoint(path)
    assert isinstance(restored, AdversarialMultVAE)
    assert restored.config == config
    assert payload["epoch"] == 3
    sample = torch.eye(9)[:2]
    model.eval()
    assert torch.allclose(model.score(sample), restored.score(sample))


def test_serving_recommend_has_no_gender_dependency() -> None:
    parameters = inspect.signature(MovieRecommender.recommend).parameters
    assert "gender" not in parameters
    assert "genders" not in parameters

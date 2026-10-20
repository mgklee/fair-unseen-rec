from __future__ import annotations

import inspect

import pytest
import torch

from fairrec.config import ModelConfig
from fairrec.model import AdversarialMultVAE, MultVAE, gradient_reverse


def test_multvae_shapes_and_deterministic_mean_path() -> None:
    model = MultVAE(ModelConfig(num_items=17, hidden_dims=(11, 7), latent_dim=5, dropout=0.0))
    interactions = torch.zeros(4, 17)
    interactions[:, :3] = 1
    output = model(interactions, sample=False)

    assert output.logits.shape == (4, 17)
    assert output.mu.shape == output.logvar.shape == output.z.shape == (4, 5)
    assert torch.equal(output.z, output.mu)
    assert torch.equal(model.score(interactions), output.logits)


def test_adversarial_model_shapes() -> None:
    model = AdversarialMultVAE(
        ModelConfig(
            num_items=13,
            hidden_dims=(9,),
            latent_dim=4,
            adversary_hidden_dims=(6,),
            dropout=0.0,
        )
    )
    output = model(torch.eye(13)[:3], grl_lambda=0.7, sample=False)
    assert output.logits.shape == (3, 13)
    assert output.z.shape == (3, 4)
    assert output.gender_logits.shape == (3, 2)


def test_gradient_reversal_identity_and_scaled_negative_gradient() -> None:
    value = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    coefficient = 0.35
    reversed_value = gradient_reverse(value, coefficient)
    assert torch.equal(reversed_value, value)
    (reversed_value * torch.tensor([2.0, 3.0, 4.0])).sum().backward()
    assert value.grad is not None
    assert torch.allclose(value.grad, torch.tensor([-0.7, -1.05, -1.4]))


def test_model_and_serving_interfaces_have_no_gender_argument() -> None:
    for method in (MultVAE.forward, MultVAE.encode, MultVAE.score):
        assert "gender" not in inspect.signature(method).parameters


def test_invalid_shape_is_rejected() -> None:
    model = MultVAE(ModelConfig(num_items=7, hidden_dims=(5,), latent_dim=3))
    with pytest.raises(ValueError, match="interactions"):
        model(torch.zeros(2, 8))


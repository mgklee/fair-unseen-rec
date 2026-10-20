"""Versioned, device-portable checkpoints for both MultVAE variants."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from .config import ModelConfig
from .model import MultVAE, build_model

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: MultVAE,
    *,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any] | None = None,
    experiment_config: Mapping[str, Any] | None = None,
    data_fingerprint: str | None = None,
) -> Path:
    """Atomically save everything needed for demographic-free inference."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model_type": model.model_type,
        "model_config": model.config.__dict__
        if hasattr(model.config, "__dict__")
        else {
            field: getattr(model.config, field)
            for field in model.config.__dataclass_fields__
        },
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
        "experiment_config": dict(experiment_config or {}),
        "data_fingerprint": data_fingerprint,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> tuple[MultVAE, dict[str, Any]]:
    """Reconstruct and strictly load a model without needing a dataset object."""

    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format_version')!r}"
        )
    for key in ("model_type", "model_config", "state_dict"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key!r}")
    config = ModelConfig.from_dict(payload["model_config"])
    model = build_model(config, str(payload["model_type"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    if eval_mode:
        model.eval()
    return model, payload


# Short alias convenient for CLI code.
load_checkpoint = load_model_checkpoint


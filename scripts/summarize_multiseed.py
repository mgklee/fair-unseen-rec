#!/usr/bin/env python3
"""Aggregate paired MultVAE vs fair-model metrics across fixed random seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = {
    "ndcg@10": ("ndcg@10",),
    "ndcg_gender_gap": ("absolute_gender_gaps", "ndcg@10"),
    "genre_exposure_l1": ("genre_exposure", "l1_between_genders"),
    "genre_residual_l1": (
        "genre_exposure",
        "preference_amplification",
        "residual_l1",
    ),
    "probe_auc_distance": (
        "posthoc_gender_probe",
        "test",
        "auc_distance_from_random",
    ),
    "coverage": ("coverage",),
}
LOWER_IS_BETTER = {
    "ndcg@10": False,
    "ndcg_gender_gap": True,
    "genre_exposure_l1": True,
    "genre_residual_l1": True,
    "probe_auc_distance": True,
    "coverage": False,
}


def nested(row: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def parse_comparison(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use SEED=PATH")
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        type=parse_comparison,
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    data_fingerprint: str | None = None
    for seed, path in args.comparison:
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = str(payload["data_fingerprint"])
        if data_fingerprint is not None and fingerprint != data_fingerprint:
            raise ValueError("comparison files use different data artifacts")
        data_fingerprint = fingerprint
        naive = payload["models"]["MultVAE"]["support_results"]
        fair = payload["models"]["Adv-MultVAE"]["support_results"]
        if set(naive) != set(fair):
            raise ValueError(f"support sizes differ in {path}")
        for support in sorted(naive, key=int):
            row: dict[str, Any] = {"seed": seed, "support_size": int(support)}
            for metric, metric_path in METRICS.items():
                baseline = nested(naive[support], metric_path)
                mitigated = nested(fair[support], metric_path)
                row[metric] = {
                    "naive": baseline,
                    "fair": mitigated,
                    "delta_fair_minus_naive": mitigated - baseline,
                }
            records.append(row)

    seeds = sorted({int(row["seed"]) for row in records})
    supports = sorted({int(row["support_size"]) for row in records})
    by_support: dict[str, Any] = {}
    for support in supports:
        selected = [row for row in records if row["support_size"] == support]
        metric_summary: dict[str, Any] = {}
        for metric in METRICS:
            naive = [row[metric]["naive"] for row in selected]
            fair = [row[metric]["fair"] for row in selected]
            delta = [row[metric]["delta_fair_minus_naive"] for row in selected]
            lower = LOWER_IS_BETTER[metric]
            wins = sum(value < 0 if lower else value > 0 for value in delta)
            metric_summary[metric] = {
                "naive": stats(naive),
                "fair": stats(fair),
                "delta_fair_minus_naive": stats(delta),
                "fair_better_runs": wins,
                "runs": len(delta),
                "lower_is_better": lower,
            }
        by_support[str(support)] = metric_summary

    macro: dict[str, Any] = {}
    for metric in METRICS:
        delta = [row[metric]["delta_fair_minus_naive"] for row in records]
        lower = LOWER_IS_BETTER[metric]
        macro[metric] = {
            "delta_fair_minus_naive": stats(delta),
            "fair_better_cases": sum(value < 0 if lower else value > 0 for value in delta),
            "cases": len(delta),
            "lower_is_better": lower,
        }

    result = {
        "protocol": {
            "paired_seeds": seeds,
            "support_sizes": supports,
            "selection": "validation only; test comparisons used for reporting, not tuning",
            "uncertainty": "sample standard deviation across three training seeds",
            "warning": (
                "Three seeds do not establish statistical significance. Lower genre "
                "exposure does not by itself imply fair utility or representation privacy."
            ),
        },
        "data_fingerprint": data_fingerprint,
        "by_support": by_support,
        "macro_over_seed_support_cases": macro,
        "paired_records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "seeds": seeds, "supports": supports}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

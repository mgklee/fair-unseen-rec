#!/usr/bin/env python3
"""Recommend ML-1M movies to a new user from liked movie IDs (no gender input)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fairrec.inference import MovieRecommender  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "data/processed/movielens_1m.npz"
    )
    parser.add_argument(
        "--like",
        "--liked-movie-id",
        dest="liked_movie_ids",
        type=int,
        action="append",
        required=True,
        help="MovieLens movie ID; repeat for every liked movie",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    recommender = MovieRecommender.from_checkpoint(
        args.checkpoint, args.data, device=args.device
    )
    recommendations = recommender.recommend(args.liked_movie_ids, k=args.top_k)
    result = {
        "input": {
            "liked_movie_ids": list(dict.fromkeys(args.liked_movie_ids)),
            "gender_supplied": False,
        },
        "recommendations": [
            {
                "rank": rank,
                "movie_id": recommendation.movie_id,
                "title": recommendation.title,
                "year": recommendation.year,
                "genres": list(recommendation.genres),
                "score": recommendation.score,
            }
            for rank, recommendation in enumerate(recommendations, start=1)
        ],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

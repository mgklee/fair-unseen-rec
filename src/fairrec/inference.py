"""Gender-free recommendation adapter for web and command-line clients."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .checkpoint import load_model_checkpoint
from .data import ML1MData, Movie, artifact_fingerprint, load_ml1m_npz
from .model import MultVAE


@dataclass(frozen=True, slots=True)
class Recommendation:
    index: int
    movie_id: int
    title: str
    year: int
    genres: tuple[str, ...]
    score: float


class MovieRecommender:
    """Serve an unseen user from liked movie IDs and nothing demographic."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        data_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        verify_fingerprint: bool = True,
    ) -> None:
        model, payload = load_model_checkpoint(checkpoint_path, device=device)
        data = load_ml1m_npz(data_path)
        if model.config.num_items != data.num_items:
            raise ValueError("checkpoint catalog size differs from the ML-1M artifact")
        expected = payload.get("data_fingerprint")
        if verify_fingerprint and expected and artifact_fingerprint(data_path) != expected:
            raise ValueError("checkpoint was trained with a different data artifact")
        self.model = model
        self.data = data
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.checkpoint_sha256 = artifact_fingerprint(self.checkpoint_path)
        self.model_type = str(payload.get("model_type", model.model_type))
        self.checkpoint_epoch = int(payload.get("epoch", -1))
        threshold = float(
            payload.get("experiment_config", {})
            .get("data", {})
            .get("positive_threshold", 4.0)
        )
        warm = data.warm_item_mask(
            split="train", positive_threshold=threshold
        )
        self.warm_item_mask = torch.as_tensor(warm, dtype=torch.bool, device=self.device)
        self._catalog_movies = data.movies
        self.movies = tuple(
            movie for movie in self._catalog_movies if bool(warm[movie.index])
        )
        self._movie_by_id = {movie.movie_id: movie for movie in self.movies}
        self._folded_titles = tuple(movie.title.casefold() for movie in self.movies)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        data_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        verify_fingerprint: bool = True,
    ) -> "MovieRecommender":
        return cls(
            checkpoint_path,
            data_path,
            device=device,
            verify_fingerprint=verify_fingerprint,
        )

    def search(self, query: str, limit: int = 20) -> list[Movie]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        needle = str(query).strip().casefold()
        if not needle:
            return list(self.movies[:limit])
        return [
            movie
            for movie, title in zip(self.movies, self._folded_titles)
            if needle in title
        ][:limit]

    def _indices_from_ids(self, movie_ids: Iterable[int]) -> list[int]:
        indices: list[int] = []
        unknown: list[int] = []
        seen: set[int] = set()
        for raw_id in movie_ids:
            movie_id = int(raw_id)
            movie = self._movie_by_id.get(movie_id)
            if movie is None:
                unknown.append(movie_id)
            elif movie.index not in seen:
                indices.append(movie.index)
                seen.add(movie.index)
        if unknown:
            raise ValueError(f"unknown MovieLens movie IDs: {unknown[:10]}")
        return indices

    @torch.inference_mode()
    def recommend(
        self,
        liked_movie_ids: Sequence[int],
        k: int = 10,
        *,
        exclude_movie_ids: Sequence[int] = (),
    ) -> list[Recommendation]:
        """Rank the catalog; gender cannot be supplied to this API by design."""

        if k <= 0:
            return []
        liked = self._indices_from_ids(liked_movie_ids)
        if not liked:
            raise ValueError("at least one liked MovieLens movie ID is required")
        excluded = set(liked)
        excluded.update(self._indices_from_ids(exclude_movie_ids))
        interactions = self.data.binary_vector(liked).unsqueeze(0).to(self.device)
        scores = self.model.score(interactions).squeeze(0)
        scores[~self.warm_item_mask] = -torch.inf
        if excluded:
            scores[torch.as_tensor(sorted(excluded), device=self.device)] = -torch.inf
        available = int(self.warm_item_mask.sum().item()) - len(excluded)
        take = min(int(k), available)
        if take <= 0:
            return []
        values, indices = torch.topk(scores, k=take)
        output: list[Recommendation] = []
        for score, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
            movie = self._catalog_movies[index]
            output.append(
                Recommendation(
                    index=movie.index,
                    movie_id=movie.movie_id,
                    title=movie.title,
                    year=movie.year,
                    genres=movie.genres,
                    score=float(score),
                )
            )
        return output


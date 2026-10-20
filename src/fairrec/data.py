"""MovieLens-1M loading and chronological unseen-user data utilities.

The artifact is a timestamp-sorted CSR-like collection of user histories.  A
user index is used only to retrieve a row; it is never passed to a model.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Sampler

GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
SPLIT_TO_INDEX = {"train": 0, "val": 1, "test": 2}
GENDER_TO_INDEX = {"M": 0, "F": 1, "U": 2}


@dataclass(frozen=True, slots=True)
class Movie:
    index: int
    movie_id: int
    title: str
    year: int
    genres: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UserHistory:
    """One timestamp-ordered row from the sparse artifact."""

    user_index: int
    user_id: int
    gender: int
    item_indices: NDArray[np.int32]
    ratings: NDArray[np.float32]
    timestamps: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ChronologicalEpisode:
    """Positive-prefix input and subsequent query events for an unseen user."""

    user_index: int
    gender: int
    support_item_indices: NDArray[np.int32]
    support_ratings: NDArray[np.float32]
    positive_support_items: NDArray[np.int32]
    query_item_indices: NDArray[np.int32]
    query_ratings: NDArray[np.float32]
    positive_query_items: NDArray[np.int32]


@dataclass(frozen=True)
class ML1MData:
    """Validated in-memory view of the processed MovieLens-1M artifact."""

    movie_ids: NDArray[np.int32]
    movie_titles: NDArray[np.str_]
    movie_years: NDArray[np.int16]
    movie_genres: NDArray[np.uint8]
    user_ids: NDArray[np.int32]
    user_genders: NDArray[np.uint8]
    user_splits: NDArray[np.uint8]
    user_ptr: NDArray[np.int64]
    rating_item_indices: NDArray[np.int32]
    rating_values: NDArray[np.float32]
    rating_timestamps: NDArray[np.int64]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        num_items = len(self.movie_ids)
        num_users = len(self.user_ids)
        num_events = len(self.rating_values)
        if num_items < 2 or num_users < 1:
            raise ValueError("artifact must contain users and at least two items")
        if len(self.movie_titles) != num_items or len(self.movie_years) != num_items:
            raise ValueError("movie metadata row counts differ")
        if self.movie_genres.ndim != 2 or self.movie_genres.shape[0] != num_items:
            raise ValueError("movie_genres must be [num_items, num_genres]")
        if len(self.user_genders) != num_users or len(self.user_splits) != num_users:
            raise ValueError("user metadata row counts differ")
        if self.user_ptr.shape != (num_users + 1,):
            raise ValueError("user_ptr must contain one offset per user plus an end")
        if self.user_ptr[0] != 0 or self.user_ptr[-1] != num_events:
            raise ValueError("user_ptr endpoints do not match rating arrays")
        if np.any(np.diff(self.user_ptr) < 0):
            raise ValueError("user_ptr must be non-decreasing")
        if len(self.rating_item_indices) != num_events or len(self.rating_timestamps) != num_events:
            raise ValueError("rating array lengths differ")
        if np.any(self.rating_item_indices < 0) or np.any(self.rating_item_indices >= num_items):
            raise ValueError("rating_item_indices contains an out-of-catalog index")
        if np.any((self.rating_values < 1) | (self.rating_values > 5)):
            raise ValueError("rating values must be in [1, 5]")
        if np.any(~np.isin(self.user_splits, tuple(SPLIT_TO_INDEX.values()))):
            raise ValueError("user_splits contains an unknown split")
        for user_index in range(num_users):
            start, stop = int(self.user_ptr[user_index]), int(self.user_ptr[user_index + 1])
            if np.any(np.diff(self.rating_timestamps[start:stop]) < 0):
                raise ValueError(f"history {user_index} is not chronological")

    @property
    def num_items(self) -> int:
        return int(len(self.movie_ids))

    @property
    def num_users(self) -> int:
        return int(len(self.user_ids))

    @property
    def movies(self) -> tuple[Movie, ...]:
        genre_names = GENRES
        return tuple(
            Movie(
                index=index,
                movie_id=int(self.movie_ids[index]),
                title=str(self.movie_titles[index]),
                year=int(self.movie_years[index]),
                genres=tuple(
                    genre_names[column]
                    if column < len(genre_names)
                    else f"genre_{column}"
                    for column in np.flatnonzero(self.movie_genres[index])
                ),
            )
            for index in range(self.num_items)
        )

    def split_users(self, split: str) -> NDArray[np.int64]:
        try:
            split_index = SPLIT_TO_INDEX[split]
        except KeyError as exc:
            raise ValueError(f"unknown split {split!r}; use train, val, or test") from exc
        return np.flatnonzero(self.user_splits == split_index).astype(np.int64)

    def history(self, user_index: int) -> UserHistory:
        if not 0 <= int(user_index) < self.num_users:
            raise IndexError(f"user index out of range: {user_index}")
        start = int(self.user_ptr[user_index])
        stop = int(self.user_ptr[user_index + 1])
        return UserHistory(
            user_index=int(user_index),
            user_id=int(self.user_ids[user_index]),
            gender=int(self.user_genders[user_index]),
            item_indices=self.rating_item_indices[start:stop],
            ratings=self.rating_values[start:stop],
            timestamps=self.rating_timestamps[start:stop],
        )

    def chronological_episode(
        self,
        user_index: int,
        support_size: int,
        *,
        positive_threshold: float = 4.0,
        item_eligible: NDArray[np.bool_] | None = None,
    ) -> ChronologicalEpisode:
        """Use the first ``support_size`` chronological positive events.

        The website asks an unseen user for movies they liked, so an implicit
        support size counts positive interactions rather than every historical
        rating. Later positive events form the query. This keeps offline
        evaluation aligned with the serving input.
        """

        if support_size < 0:
            raise ValueError("support_size must be non-negative")
        history = self.history(user_index)
        positive = history.ratings >= positive_threshold
        eligibility = None
        if item_eligible is not None:
            eligibility = np.asarray(item_eligible, dtype=bool).reshape(-1)
            if eligibility.shape != (self.num_items,):
                raise ValueError("item_eligible must have one value per catalog item")
        support_candidates = positive.copy()
        if eligibility is not None:
            support_candidates &= eligibility[history.item_indices]
        support_positions = np.flatnonzero(support_candidates)[: int(support_size)]
        support_items = history.item_indices[support_positions]
        support_ratings = history.ratings[support_positions]
        # Query starts after the final selected support event but retains cold
        # positives so evaluation can explicitly count and exclude them.
        boundary = int(support_positions[-1]) + 1 if support_positions.size else 0
        query_mask = positive[boundary:]
        query_items = history.item_indices[boundary:][query_mask]
        query_ratings = history.ratings[boundary:][query_mask]
        return ChronologicalEpisode(
            user_index=int(user_index),
            gender=history.gender,
            support_item_indices=support_items,
            support_ratings=support_ratings,
            positive_support_items=support_items,
            query_item_indices=query_items,
            query_ratings=query_ratings,
            positive_query_items=query_items,
        )

    def warm_item_mask(
        self,
        *,
        split: str = "train",
        positive_threshold: float = 4.0,
    ) -> NDArray[np.bool_]:
        """Return items with at least one positive interaction in ``split`` users."""

        mask = np.zeros(self.num_items, dtype=bool)
        for user_index in self.split_users(split):
            history = self.history(int(user_index))
            mask[history.item_indices[history.ratings >= positive_threshold]] = True
        return mask

    def binary_vector(self, item_indices: Sequence[int] | NDArray[np.integer[Any]]) -> torch.Tensor:
        """Create the sole model input representation: a binary catalog vector."""

        output = torch.zeros(self.num_items, dtype=torch.float32)
        indices = np.asarray(item_indices, dtype=np.int64)
        if indices.size:
            if np.any(indices < 0) or np.any(indices >= self.num_items):
                raise IndexError("item index out of catalog range")
            output[torch.as_tensor(np.unique(indices), dtype=torch.long)] = 1.0
        return output


def load_ml1m_npz(path: str | Path) -> ML1MData:
    """Load the standalone NPZ schema also used by the predecessor project."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "movie_ids",
        "movie_titles",
        "movie_years",
        "movie_genres",
        "user_ids",
        "user_genders",
        "user_splits",
        "user_ptr",
        "rating_item_indices",
        "rating_values",
        "rating_timestamps",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"processed artifact is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    return ML1MData(
        movie_ids=arrays["movie_ids"].astype(np.int32, copy=False),
        movie_titles=arrays["movie_titles"].astype(np.str_, copy=False),
        movie_years=arrays["movie_years"].astype(np.int16, copy=False),
        movie_genres=arrays["movie_genres"].astype(np.uint8, copy=False),
        user_ids=arrays["user_ids"].astype(np.int32, copy=False),
        user_genders=arrays["user_genders"].astype(np.uint8, copy=False),
        user_splits=arrays["user_splits"].astype(np.uint8, copy=False),
        user_ptr=arrays["user_ptr"].astype(np.int64, copy=False),
        rating_item_indices=arrays["rating_item_indices"].astype(np.int32, copy=False),
        rating_values=arrays["rating_values"].astype(np.float32, copy=False),
        rating_timestamps=arrays["rating_timestamps"].astype(np.int64, copy=False),
        source_path=source,
    )


def artifact_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UserInteractionDataset(Dataset[dict[str, torch.Tensor]]):
    """Dense-on-access binary vectors; gender is emitted only for training/audit."""

    def __init__(
        self,
        data: ML1MData,
        *,
        split: str | None = None,
        user_indices: Sequence[int] | NDArray[np.integer[Any]] | None = None,
        positive_threshold: float = 4.0,
        prefix_size: int | None = None,
        target: Literal["all", "future"] = "all",
        random_support_sizes: Sequence[int] | None = None,
        full_profile_probability: float = 0.0,
        seed: int = 42,
    ) -> None:
        if split is not None and user_indices is not None:
            raise ValueError("provide split or user_indices, not both")
        if target not in {"all", "future"}:
            raise ValueError("target must be all or future")
        if prefix_size is not None and prefix_size < 0:
            raise ValueError("prefix_size must be non-negative")
        if prefix_size is not None and random_support_sizes is not None:
            raise ValueError("prefix_size and random_support_sizes are mutually exclusive")
        if random_support_sizes is not None and (
            not random_support_sizes or any(int(size) <= 0 for size in random_support_sizes)
        ):
            raise ValueError("random_support_sizes must contain positive counts")
        if not 0.0 <= full_profile_probability <= 1.0:
            raise ValueError("full_profile_probability must be in [0, 1]")
        self.data = data
        self.user_indices = (
            data.split_users(split)
            if split is not None
            else np.asarray(
                np.arange(data.num_users) if user_indices is None else user_indices,
                dtype=np.int64,
            )
        )
        if np.any(self.user_indices < 0) or np.any(self.user_indices >= data.num_users):
            raise IndexError("dataset user index out of range")
        self.positive_threshold = float(positive_threshold)
        self.prefix_size = prefix_size
        self.target_mode = target
        self.random_support_sizes = (
            None
            if random_support_sizes is None
            else tuple(int(size) for size in random_support_sizes)
        )
        self.full_profile_probability = float(full_profile_probability)
        self.seed = int(seed)
        self.epoch = 0
        self.genders = data.user_genders[self.user_indices].astype(np.int64, copy=False)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic new few-shot view for each training epoch."""

        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.user_indices))

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        user_index = int(self.user_indices[position])
        history = self.data.history(user_index)
        all_positive = history.item_indices[history.ratings >= self.positive_threshold]
        if self.random_support_sizes is not None:
            rng = np.random.default_rng(
                np.random.SeedSequence((self.seed, self.epoch, user_index))
            )
            use_full = rng.random() < self.full_profile_probability
            if use_full or all_positive.size == 0:
                input_items = all_positive
            else:
                support_size = int(rng.choice(self.random_support_sizes))
                take = min(support_size, int(all_positive.size))
                selected = rng.choice(all_positive.size, size=take, replace=False)
                input_items = all_positive[np.sort(selected)]
            # Denoising target stays the complete profile, teaching the ID-free
            # encoder to infer held-out tastes from a handful of known likes.
            target_items = all_positive
        elif self.prefix_size is None:
            input_items = all_positive
            target_items = all_positive
        else:
            episode = self.data.chronological_episode(
                user_index,
                self.prefix_size,
                positive_threshold=self.positive_threshold,
            )
            input_items = episode.positive_support_items
            target_items = (
                episode.positive_query_items
                if self.target_mode == "future"
                else all_positive
            )
        return {
            "input": self.data.binary_vector(input_items),
            "target": self.data.binary_vector(target_items),
            "gender": torch.tensor(history.gender, dtype=torch.long),
            # Metadata for audit/debug only. Model APIs deliberately reject it.
            "user_index": torch.tensor(user_index, dtype=torch.long),
        }


class BalancedGenderBatchSampler(Sampler[list[int]]):
    """Yield exactly half M/F dataset positions, oversampling when necessary."""

    def __init__(
        self,
        genders: Sequence[int] | NDArray[np.integer[Any]],
        batch_size: int,
        *,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        labels = np.asarray(genders, dtype=np.int64)
        if labels.ndim != 1:
            raise ValueError("genders must be one-dimensional")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("balanced batch_size must be an even integer >= 2")
        self.groups = (np.flatnonzero(labels == 0), np.flatnonzero(labels == 1))
        if any(len(group) == 0 for group in self.groups):
            raise ValueError("both M=0 and F=1 examples are required")
        self.size = 2 * max(len(group) for group in self.groups)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        operation = math.floor if self.drop_last else math.ceil
        return operation(self.size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        half = self.batch_size // 2
        draws_per_group = len(self) * half

        def shuffled_cycles(group: NDArray[np.int64]) -> NDArray[np.int64]:
            pieces: list[NDArray[np.int64]] = []
            remaining = draws_per_group
            while remaining:
                cycle = rng.permutation(group)
                take = min(remaining, len(cycle))
                pieces.append(cycle[:take])
                remaining -= take
            return np.concatenate(pieces)

        streams = [shuffled_cycles(group) for group in self.groups]
        for batch_index in range(len(self)):
            start = batch_index * half
            stop = start + half
            batch = np.concatenate([stream[start:stop] for stream in streams])
            rng.shuffle(batch)
            yield [int(index) for index in batch]

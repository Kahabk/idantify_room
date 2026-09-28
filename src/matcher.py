from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .room_database import RoomDatabase
from .utils import normalize_rows


@dataclass(frozen=True)
class ReferenceMatch:
    label: str
    similarity: float
    metadata: dict


@dataclass(frozen=True)
class MatchResult:
    label: str
    confidence: float
    best_similarity: float
    score_margin: float
    room_scores: dict[str, float]
    top_matches: list[ReferenceMatch]


class RoomMatcher:
    """Aggregate frame neighbors and optionally room centroids."""

    def __init__(
        self,
        database: RoomDatabase,
        top_k: int = 10,
        room_match_count: int = 3,
        min_similarity: float = 0.60,
        min_score_margin: float = 0.02,
        prototype_weight: float = 0.20,
        match_mode: str = "hybrid",
    ) -> None:
        if top_k < 1 or room_match_count < 1:
            raise ValueError("top_k and room_match_count must be positive")
        if not 0.0 <= prototype_weight <= 1.0:
            raise ValueError("prototype_weight must be in [0, 1]")
        if match_mode not in {"frames", "prototypes", "hybrid"}:
            raise ValueError("match_mode must be frames, prototypes, or hybrid")
        self.database = database
        self.top_k = top_k
        self.room_match_count = room_match_count
        self.min_similarity = min_similarity
        self.min_score_margin = min_score_margin
        self.prototype_weight = prototype_weight
        self.match_mode = match_mode

    def match(self, embedding: np.ndarray) -> MatchResult:
        query = normalize_rows(embedding)
        similarities, indices = self.database.search(query, self.top_k)
        grouped: dict[str, list[float]] = {}
        top_matches: list[ReferenceMatch] = []
        for similarity, index in zip(similarities, indices):
            if index < 0:
                continue
            label = self.database.labels[int(index)]
            score = float(similarity)
            grouped.setdefault(label, []).append(score)
            top_matches.append(ReferenceMatch(label, score, self.database.metadata[int(index)]))

        frame_scores = {
            label: float(np.mean(scores[: self.room_match_count]))
            for label, scores in grouped.items()
        }
        centroid_values = (query @ self.database.prototypes.T)[0]
        prototype_scores = {
            label: float(centroid_values[i])
            for i, label in enumerate(self.database.prototype_labels)
        }
        if self.match_mode == "frames":
            room_scores = frame_scores
        elif self.match_mode == "prototypes":
            room_scores = prototype_scores
        else:
            room_scores = {
                label: (1.0 - self.prototype_weight) * score
                + self.prototype_weight * prototype_scores[label]
                for label, score in frame_scores.items()
            }

        ranked = sorted(room_scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score
        accepted = best_score >= self.min_similarity and margin >= self.min_score_margin
        label = best_label if accepted else "unknown"
        # Known confidence is its cosine score. UNKNOWN confidence increases as
        # the best known-room similarity falls away from the threshold.
        confidence = float(np.clip(best_score if accepted else 1.0 - best_score, 0.0, 1.0))
        return MatchResult(
            label=label,
            confidence=confidence,
            best_similarity=float(best_score),
            score_margin=float(margin),
            room_scores=room_scores,
            top_matches=top_matches,
        )


from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)
LocationChangedCallback = Callable[[str, str, float], None]


@dataclass(frozen=True)
class StablePrediction:
    location: str
    confidence: float
    window_winner: str


class TemporalLocationFilter:
    """Confidence-weighted sliding vote with confirmed state transitions."""

    def __init__(
        self,
        window_size: int = 15,
        switch_confirmation_frames: int = 5,
        unknown_confirmation_frames: int | None = None,
    ) -> None:
        if window_size < 1 or switch_confirmation_frames < 1:
            raise ValueError("Temporal filter sizes must be positive")
        self.history: deque[tuple[str, float]] = deque(maxlen=window_size)
        self.switch_confirmation_frames = switch_confirmation_frames
        self.unknown_confirmation_frames = (
            unknown_confirmation_frames
            if unknown_confirmation_frames is not None
            else switch_confirmation_frames
        )
        if self.unknown_confirmation_frames < 1:
            raise ValueError("unknown_confirmation_frames must be positive")
        self.current_location = "unknown"
        self.current_confidence = 0.0
        self._candidate = "unknown"
        self._candidate_wins = 0
        self._callbacks: list[LocationChangedCallback] = []

    def add_callback(self, callback: LocationChangedCallback) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: LocationChangedCallback) -> None:
        self._callbacks.remove(callback)

    def update(self, label: str, confidence: float) -> StablePrediction:
        confidence = max(0.0, min(1.0, float(confidence)))
        self.history.append((label, confidence))
        weights: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for observed_label, observed_confidence in self.history:
            weights[observed_label] += max(0.25, observed_confidence)
            counts[observed_label] += 1
        winner = max(weights, key=weights.get)
        winner_confidence = weights[winner] / counts[winner]

        if winner == self.current_location:
            self._candidate = winner
            self._candidate_wins = 0
            self.current_confidence = winner_confidence
        else:
            if winner == self._candidate:
                self._candidate_wins += 1
            else:
                self._candidate = winner
                self._candidate_wins = 1
            required_wins = (
                self.unknown_confirmation_frames
                if winner == "unknown"
                else self.switch_confirmation_frames
            )
            if self._candidate_wins >= required_wins:
                old_location = self.current_location
                self.current_location = winner
                self.current_confidence = winner_confidence
                self._candidate_wins = 0
                for callback in tuple(self._callbacks):
                    try:
                        callback(old_location, winner, winner_confidence)
                    except Exception:
                        LOGGER.exception("Location-change callback failed")
        return StablePrediction(self.current_location, self.current_confidence, winner)

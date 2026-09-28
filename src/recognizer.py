from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import config
from .matcher import MatchResult, RoomMatcher
from .room_database import RoomDatabase
from .temporal_filter import StablePrediction, TemporalLocationFilter
from .vision_encoder import VisionEncoder

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecognitionState:
    location: str = "unknown"
    confidence: float = 0.0
    inference_ms: float = 0.0
    match: MatchResult | None = None


class RoomRecognizer:
    """Programmatic API for synchronous, stateful room recognition."""

    def __init__(
        self,
        database_dir: str | Path = config.DATABASE_DIR,
        on_location_changed: Callable[[str, str, float], None] | None = None,
    ) -> None:
        self.database = RoomDatabase.load(Path(database_dir))
        model_name = str(self.database.manifest["model_name"])
        if model_name != config.MODEL_NAME:
            LOGGER.warning(
                "Database uses %s (config requests %s); using database model.",
                model_name,
                config.MODEL_NAME,
            )
        self.encoder = VisionEncoder(model_name, config.DEVICE)
        self.matcher = RoomMatcher(
            self.database,
            top_k=config.TOP_K,
            room_match_count=config.ROOM_MATCH_COUNT,
            min_similarity=config.MIN_SIMILARITY,
            min_score_margin=config.MIN_SCORE_MARGIN,
            prototype_weight=config.PROTOTYPE_WEIGHT,
            match_mode=config.MATCH_MODE,
        )
        self.temporal_filter = TemporalLocationFilter(
            config.PREDICTION_WINDOW,
            config.SWITCH_CONFIRMATION_FRAMES,
            config.UNKNOWN_CONFIRMATION_FRAMES,
        )
        if on_location_changed is not None:
            self.temporal_filter.add_callback(on_location_changed)
        self._state = RecognitionState()
        self._lock = threading.Lock()

    def add_location_changed_callback(
        self, callback: Callable[[str, str, float], None]
    ) -> None:
        self.temporal_filter.add_callback(callback)

    def remove_location_changed_callback(
        self, callback: Callable[[str, str, float], None]
    ) -> None:
        self.temporal_filter.remove_callback(callback)

    def process_frame(self, bgr_frame: np.ndarray) -> RecognitionState:
        started = time.perf_counter()
        embedding = self.encoder.encode([bgr_frame], batch_size=1)[0]
        match = self.matcher.match(embedding)
        stable: StablePrediction = self.temporal_filter.update(match.label, match.confidence)
        state = RecognitionState(
            location=stable.location,
            confidence=stable.confidence,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            match=match,
        )
        with self._lock:
            self._state = state
        return state

    def get_current_location(self) -> str:
        with self._lock:
            return self._state.location

    def get_current_confidence(self) -> float:
        with self._lock:
            return self._state.confidence

    def get_state(self) -> RecognitionState:
        with self._lock:
            return self._state

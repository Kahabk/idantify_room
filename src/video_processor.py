"""Reference-video discovery, sampling, and frame-quality filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class SampledFrame:
    frame: np.ndarray
    label: str
    video_path: Path
    frame_number: int
    timestamp_seconds: float
    blur_score: float
    brightness: float


class VideoProcessor:
    def __init__(
        self,
        sample_interval_seconds: float,
        min_blur_score: float,
        min_brightness: float,
        max_dark_pixel_ratio: float,
        dark_pixel_threshold: int,
        duplicate_similarity: float,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be greater than zero")
        self.sample_interval_seconds = sample_interval_seconds
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_dark_pixel_ratio = max_dark_pixel_ratio
        self.dark_pixel_threshold = dark_pixel_threshold
        self.duplicate_similarity = duplicate_similarity

    @staticmethod
    def discover_videos(dataset_dir: Path, extensions: set[str]) -> list[tuple[str, Path]]:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        videos: list[tuple[str, Path]] = []
        for path in sorted(dataset_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                relative = path.relative_to(dataset_dir)
                if len(relative.parts) < 2:
                    print(f"Skipping {path}: videos must be inside a label directory")
                    continue
                label = path.parent.name.strip().lower().replace(" ", "_")
                videos.append((label, path))
        return videos

    @staticmethod
    def _fingerprint(gray: np.ndarray) -> np.ndarray:
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        small -= small.mean()
        norm = float(np.linalg.norm(small))
        return small.reshape(-1) / max(norm, 1e-6)

    def iter_useful_frames(self, video_path: Path, label: str) -> Iterator[SampledFrame]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        sample_step = max(1, int(round(fps * self.sample_interval_seconds)))
        previous_fingerprint: np.ndarray | None = None
        frame_number = -1

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number % sample_step != 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                dark_ratio = float(np.mean(gray < self.dark_pixel_threshold))
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if brightness < self.min_brightness or dark_ratio > self.max_dark_pixel_ratio:
                    continue
                if blur_score < self.min_blur_score:
                    continue

                fingerprint = self._fingerprint(gray)
                if previous_fingerprint is not None:
                    similarity = float(fingerprint @ previous_fingerprint)
                    if similarity >= self.duplicate_similarity:
                        continue
                previous_fingerprint = fingerprint
                yield SampledFrame(
                    frame=frame,
                    label=label,
                    video_path=video_path,
                    frame_number=frame_number,
                    timestamp_seconds=frame_number / fps,
                    blur_score=blur_score,
                    brightness=brightness,
                )
        finally:
            capture.release()

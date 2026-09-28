#!/usr/bin/env python3
"""Build the persistent reference-frame embedding database."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from src.room_database import RoomDatabase
from src.video_processor import SampledFrame, VideoProcessor
from src.vision_encoder import VisionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output", type=Path, default=config.DATABASE_DIR)
    parser.add_argument("--sample-interval", type=float, default=config.FRAME_SAMPLE_INTERVAL)
    parser.add_argument("--batch-size", type=int, default=config.INFERENCE_BATCH_SIZE)
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--device", default=config.DEVICE)
    return parser.parse_args()


def build_database(args: argparse.Namespace) -> RoomDatabase:
    processor = VideoProcessor(
        sample_interval_seconds=args.sample_interval,
        min_blur_score=config.MIN_BLUR_SCORE,
        min_brightness=config.MIN_BRIGHTNESS,
        max_dark_pixel_ratio=config.MAX_DARK_PIXEL_RATIO,
        dark_pixel_threshold=config.DARK_PIXEL_THRESHOLD,
        duplicate_similarity=config.DUPLICATE_SIMILARITY,
    )
    videos = processor.discover_videos(args.dataset, config.VIDEO_EXTENSIONS)
    if not videos:
        raise RuntimeError(f"No labeled videos found under {args.dataset}")

    print(f"Found {len(videos)} videos across {len(set(label for label, _ in videos))} rooms")
    encoder = VisionEncoder(args.model, args.device)
    embeddings = []
    labels: list[str] = []
    metadata: list[dict] = []
    batch: list[SampledFrame] = []

    def flush_batch() -> None:
        if not batch:
            return
        vectors = encoder.encode([sample.frame for sample in batch])
        embeddings.extend(vectors)
        for sample in batch:
            labels.append(sample.label)
            metadata.append(
                {
                    "room_label": sample.label,
                    "video_filename": sample.video_path.name,
                    "video_path": str(sample.video_path.relative_to(args.dataset)),
                    "frame_number": sample.frame_number,
                    "timestamp_seconds": round(sample.timestamp_seconds, 3),
                    "blur_score": round(sample.blur_score, 2),
                    "brightness": round(sample.brightness, 2),
                }
            )
        batch.clear()

    for number, (label, video_path) in enumerate(videos, start=1):
        before = len(embeddings) + len(batch)
        print(f"[{number}/{len(videos)}] {label}: {video_path.name}")
        try:
            for sample in processor.iter_useful_frames(video_path, label):
                batch.append(sample)
                if len(batch) >= args.batch_size:
                    flush_batch()
        except RuntimeError as exc:
            print(f"  Warning: {exc}")
            continue
        after = len(embeddings) + len(batch)
        print(f"  accepted {after - before} frames")
    flush_batch()

    if not embeddings:
        raise RuntimeError("No frames passed the quality filters; relax thresholds in config.py")
    database = RoomDatabase.create(
        embeddings,
        labels,
        metadata,
        model_name=args.model,
        build_settings={
            "sample_interval_seconds": args.sample_interval,
            "min_blur_score": config.MIN_BLUR_SCORE,
            "min_brightness": config.MIN_BRIGHTNESS,
            "duplicate_similarity": config.DUPLICATE_SIMILARITY,
        },
    )
    database.save(args.output)
    print(f"Saved {len(labels)} embeddings for {len(database.prototype_labels)} rooms to {args.output}")
    return database


if __name__ == "__main__":
    build_database(parse_args())

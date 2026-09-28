#!/usr/bin/env python3
"""Recognize indoor rooms from a live OpenCV camera stream."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2

import config
from src.recognizer import RecognitionState, RoomRecognizer
from src.utils import format_label


class AsyncFrameRecognizer:
    """Run inference off the display thread and discard stale queued frames."""

    def __init__(self, recognizer: RoomRecognizer) -> None:
        self.recognizer = recognizer
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._run, name="room-inference", daemon=True)
        self._worker.start()

    def submit(self, frame) -> None:
        try:
            self._frames.put_nowait(frame.copy())
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(frame.copy())
            except queue.Full:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.recognizer.process_frame(frame)
            except BaseException as exc:
                self._error = exc
                self._stop.set()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("Background room inference failed") from self._error

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=config.CAMERA_ID)
    parser.add_argument("--database", type=Path, default=config.DATABASE_DIR)
    return parser.parse_args()


def draw_overlay(frame, state: RecognitionState, debug: bool, fps: float) -> None:
    color = (0, 190, 0) if state.location != "unknown" else (0, 165, 255)
    cv2.rectangle(frame, (12, 12), (620, 92), (15, 15, 15), -1)
    cv2.putText(frame, f"Current Location: {format_label(state.location)}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Confidence: {state.confidence:.0%}", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
    if debug and state.match:
        result = state.match
        lines = [
            f"Raw: {format_label(result.label)}  similarity={result.best_similarity:.3f}  margin={result.score_margin:.3f}",
            "Top matches:",
        ]
        lines.extend(f"  {format_label(match.label):16s} {match.similarity:.3f}" for match in result.top_matches[:5])
        lines.append(f"FPS: {fps:.1f}    Inference: {state.inference_ms:.1f} ms")
        cv2.rectangle(frame, (12, 105), (620, 122 + 27 * len(lines)), (15, 15, 15), -1)
        for row, line in enumerate(lines):
            cv2.putText(frame, line, (25, 130 + row * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, "q: quit   d: debug", (20, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)


def run_camera(args: argparse.Namespace) -> None:
    def on_location_changed(old_location: str, new_location: str, confidence: float) -> None:
        print(f"Location changed: {old_location} -> {new_location} ({confidence:.1%})")

    recognizer = RoomRecognizer(args.database, on_location_changed=on_location_changed)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")
    async_recognizer = AsyncFrameRecognizer(recognizer)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    debug = False
    frame_number = 0
    fps = 0.0
    fps_count = 0
    fps_started = time.perf_counter()
    state = recognizer.get_state()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed")
                break
            frame_number += 1
            fps_count += 1
            if frame_number == 1 or frame_number % config.PROCESS_EVERY_N_FRAMES == 0:
                async_recognizer.submit(frame)
            async_recognizer.check_error()
            state = recognizer.get_state()
            elapsed = time.perf_counter() - fps_started
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_started = time.perf_counter()
            if config.DISPLAY_WIDTH and frame.shape[1] > config.DISPLAY_WIDTH:
                scale = config.DISPLAY_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            draw_overlay(frame, state, debug, fps)
            cv2.imshow(config.WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug = not debug
    finally:
        async_recognizer.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera(parse_args())

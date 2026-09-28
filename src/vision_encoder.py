"""Pretrained DINOv2 image encoder."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class VisionEncoder:
    """Generate L2-normalized DINOv2 embeddings from OpenCV BGR frames."""

    def __init__(self, model_name: str, device: str) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dimension = int(self.model.config.hidden_size)

    def encode(
        self, bgr_frames: Sequence[np.ndarray], batch_size: int | None = None
    ) -> np.ndarray:
        if not bgr_frames:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        chunk_size = batch_size or len(bgr_frames)
        if chunk_size < 1:
            raise ValueError("batch_size must be positive")
        result: list[np.ndarray] = []
        for start in range(0, len(bgr_frames), chunk_size):
            chunk = bgr_frames[start : start + chunk_size]
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in chunk]
            inputs = self.processor(images=rgb_frames, return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            result.append(embeddings.float().cpu().numpy())
        return np.concatenate(result, axis=0)

    def encode_one(self, bgr_frame: np.ndarray) -> np.ndarray:
        return self.encode([bgr_frame])[0]

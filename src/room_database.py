from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import load_json, normalize_rows, save_json


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("FAISS is missing. Run: pip install -r requirements.txt") from exc
    return faiss


class RoomDatabase:
    """Persistent normalized embeddings plus exact inner-product FAISS index."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: Sequence[str],
        metadata: Sequence[dict[str, Any]],
        index: Any,
        prototypes: np.ndarray,
        prototype_labels: Sequence[str],
        manifest: dict[str, Any],
    ) -> None:
        self.embeddings = normalize_rows(embeddings)
        self.labels = list(labels)
        self.metadata = list(metadata)
        self.index = index
        self.prototypes = normalize_rows(prototypes)
        self.prototype_labels = list(prototype_labels)
        self.manifest = manifest
        if not (len(self.embeddings) == len(self.labels) == len(self.metadata)):
            raise ValueError("Database embeddings, labels, and metadata have different lengths")
        if len(self.prototypes) != len(self.prototype_labels):
            raise ValueError("Prototype embeddings and labels have different lengths")
        if self.embeddings.shape[1] != self.prototypes.shape[1]:
            raise ValueError("Frame and prototype embedding dimensions differ")

    @classmethod
    def create(
        cls,
        embeddings: np.ndarray,
        labels: Sequence[str],
        metadata: Sequence[dict[str, Any]],
        model_name: str,
        build_settings: dict[str, Any] | None = None,
    ) -> "RoomDatabase":
        vectors = normalize_rows(embeddings)
        if len(vectors) == 0:
            raise ValueError("Cannot create an empty room database")
        if len(vectors) != len(labels) or len(vectors) != len(metadata):
            raise ValueError("Each embedding must have one label and metadata record")

        faiss = _faiss()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors))

        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for vector, label in zip(vectors, labels):
            grouped[str(label)].append(vector)
        prototype_labels = sorted(grouped)
        prototypes = normalize_rows(
            np.stack([np.mean(grouped[label], axis=0) for label in prototype_labels])
        )
        manifest = {
            "format_version": cls.FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "embedding_dimension": int(vectors.shape[1]),
            "embedding_count": int(len(vectors)),
            "room_count": int(len(prototype_labels)),
            "rooms": {label: len(grouped[label]) for label in prototype_labels},
            "build_settings": build_settings or {},
        }
        return cls(vectors, labels, metadata, index, prototypes, prototype_labels, manifest)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        np.save(directory / "prototypes.npy", self.prototypes)
        save_json(directory / "labels.json", self.labels)
        save_json(directory / "metadata.json", self.metadata)
        save_json(directory / "prototype_labels.json", self.prototype_labels)
        save_json(directory / "manifest.json", self.manifest)
        _faiss().write_index(self.index, str(directory / "room_index.faiss"))

    @classmethod
    def load(cls, directory: Path) -> "RoomDatabase":
        required = [
            "embeddings.npy",
            "labels.json",
            "metadata.json",
            "prototypes.npy",
            "prototype_labels.json",
            "manifest.json",
            "room_index.faiss",
        ]
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Room database is incomplete in {directory}; missing: {', '.join(missing)}. "
                "Run build_room_database.py first."
            )
        manifest = load_json(directory / "manifest.json")
        if manifest.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported room database format; rebuild the database")
        embeddings = np.load(directory / "embeddings.npy", allow_pickle=False)
        prototypes = np.load(directory / "prototypes.npy", allow_pickle=False)
        index = _faiss().read_index(str(directory / "room_index.faiss"))
        if index.ntotal != len(embeddings):
            raise ValueError("FAISS index size does not match embeddings; rebuild the database")
        return cls(
            embeddings=embeddings,
            labels=load_json(directory / "labels.json"),
            metadata=load_json(directory / "metadata.json"),
            index=index,
            prototypes=prototypes,
            prototype_labels=load_json(directory / "prototype_labels.json"),
            manifest=manifest,
        )

    def search(self, embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query = normalize_rows(embedding)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match the database")
        k = min(max(1, int(top_k)), len(self.embeddings))
        similarities, indices = self.index.search(query, k)
        return similarities[0], indices[0]

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.matcher import RoomMatcher
from src.room_database import RoomDatabase
from src.temporal_filter import TemporalLocationFilter


class CoreTests(unittest.TestCase):
    def make_database(self):
        embeddings = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0], [0.2, 0.98]], dtype=np.float32)
        labels = ["hall", "hall", "kitchen", "kitchen"]
        metadata = [{"video_filename": "x.mp4", "timestamp_seconds": i} for i in range(4)]
        return RoomDatabase.create(embeddings, labels, metadata, model_name="test/model")

    def test_match_and_unknown(self):
        matcher = RoomMatcher(
            self.make_database(),
            top_k=4,
            room_match_count=2,
            min_similarity=0.65,
            min_score_margin=0.05,
            prototype_weight=0.25,
        )
        self.assertEqual(matcher.match(np.array([1.0, 0.0])).label, "hall")
        self.assertEqual(matcher.match(np.array([-1.0, -1.0])).label, "unknown")

    def test_temporal_switch_requires_confirmation(self):
        temporal = TemporalLocationFilter(window_size=5, switch_confirmation_frames=3)
        events = []
        temporal.add_callback(lambda old, new, confidence: events.append((old, new)))
        temporal.update("hall", 0.9)
        temporal.update("hall", 0.9)
        temporal.update("hall", 0.9)
        self.assertEqual(events, [("unknown", "hall")])
        self.assertEqual(temporal.current_location, "hall")
        temporal.update("kitchen", 0.95)
        self.assertEqual(temporal.current_location, "hall")

    def test_single_bad_frame_does_not_switch(self):
        temporal = TemporalLocationFilter(window_size=7, switch_confirmation_frames=2)
        for _ in range(4):
            temporal.update("hall", 0.9)
        temporal.update("kitchen", 0.99)
        temporal.update("hall", 0.9)
        self.assertEqual(temporal.current_location, "hall")

    def test_unknown_requires_longer_confirmation(self):
        temporal = TemporalLocationFilter(
            window_size=3,
            switch_confirmation_frames=1,
            unknown_confirmation_frames=3,
        )
        temporal.update("hall", 0.9)
        for _ in range(3):
            temporal.update("unknown", 0.8)
        self.assertEqual(temporal.current_location, "hall")
        temporal.update("unknown", 0.8)
        self.assertEqual(temporal.current_location, "unknown")

    def test_database_round_trip(self):
        database = self.make_database()
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            database.save(path)
            loaded = RoomDatabase.load(path)
            similarities, indices = loaded.search(np.array([1.0, 0.0]), 2)
            self.assertEqual(loaded.manifest["model_name"], "test/model")
            self.assertEqual(loaded.labels[int(indices[0])], "hall")
            self.assertGreater(float(similarities[0]), 0.99)


if __name__ == "__main__":
    unittest.main()

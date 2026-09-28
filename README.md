# Real-time indoor room recognition

This project indexes labeled reference videos and recognizes the current room from a live camera without training a custom network. It uses the pretrained `facebook/dinov2-small` visual backbone, cosine nearest-neighbor search, room prototypes, unknown-location rejection, and temporally confirmed location changes.

## Why DINOv2

DINOv2 is a strong general-purpose visual feature extractor designed for direct nearest-neighbor use. Unlike a text-aligned model, this task mainly needs visual instance/place similarity (layout, furniture, texture, and viewpoint). The small backbone is the default because it offers a useful latency/quality balance. Change `MODEL_NAME` in `config.py` to `facebook/dinov2-base` for more accuracy if the hardware can support it, then rebuild the database.

## Dataset

Place reference videos below `dataset/`, one folder per room. The immediate parent folder is the normalized lowercase label:

```text
dataset/
├── kitchen/
│   ├── kitchen_01.mp4
│   └── kitchen_02.mp4
├── bedroom/
│   └── bedroom_01.mp4
└── hall/
    └── hall_01.mp4
```

Record slow sweeps from multiple positions, heights, viewing directions, and lighting conditions. Avoid spending most of a video pointed at a blank wall. More varied references are generally more valuable than many nearly identical frames.

## Ubuntu installation

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch's generic wheel supports CPU. For a CUDA-specific PyTorch install, use the command generated for your installed CUDA version at https://pytorch.org/get-started/locally/ before installing the remaining requirements.

The first database build downloads the pretrained model weights from Hugging Face and caches them locally.

## Build and run

```bash
python build_room_database.py
python realtime_room_recognition.py
```

Optional overrides:

```bash
python build_room_database.py --dataset /path/to/dataset --sample-interval 0.75
python realtime_room_recognition.py --camera 1 --database /path/to/room_database
```

Press `d` for detailed matches/timing and `q` to quit. Inference runs on a
latest-frame background worker, so the display remains responsive and stale
queued frames are dropped instead of accumulating latency.

The database contains `embeddings.npy`, `labels.json`, `metadata.json`,
`prototypes.npy`, `prototype_labels.json`, `manifest.json`, and
`room_index.faiss`. Individual embeddings preserve viewpoints; prototypes add
room-level evidence. Normalized-vector inner product in FAISS is cosine
similarity here.

To compare retrieval strategies without rebuilding, set `MATCH_MODE` in
`config.py` (or the `ROOM_MATCH_MODE` environment variable) to `frames`,
`prototypes`, or `hybrid`. The hybrid default combines viewpoint-specific
neighbors with the corresponding room centroid.

## Robot integration

Import the stateful class and feed it OpenCV BGR frames at the cadence you choose:

```python
from src.recognizer import RoomRecognizer

recognizer = RoomRecognizer()

def on_location_changed(old_location, new_location, confidence):
    print(old_location, new_location, confidence)

recognizer.add_location_changed_callback(on_location_changed)
state = recognizer.process_frame(frame)
location = recognizer.get_current_location()  # hall, kitchen, bedroom, or unknown
```

`state.match` contains the raw per-frame decision and top matches.
`get_current_location()` returns the temporally filtered state.

## Calibration

Start with the defaults, then inspect debug scores in known rooms and in genuinely unseen areas:

- Raise `MIN_SIMILARITY` if unseen places are accepted as known; lower it if known rooms become `unknown`.
- Raise `MIN_SCORE_MARGIN` if visually similar rooms are confused.
- Increase `SWITCH_CONFIRMATION_FRAMES` or `PREDICTION_WINDOW` for more stability at the cost of slower transitions.
- Increase `UNKNOWN_CONFIRMATION_FRAMES` if short blurred/occluded sequences still make the stable location become `UNKNOWN`.
- Lower `PROCESS_EVERY_N_FRAMES` for quicker reactions, or raise it for lower compute use.
- If too many reference frames are discarded, lower `MIN_BLUR_SCORE` or raise `DUPLICATE_SIMILARITY`.

Thresholds are environment-specific. Validate them with separate walk-through videos that were not used to build the database.

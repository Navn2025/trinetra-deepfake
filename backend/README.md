# Trinetra Backend

FastAPI backend that powers the Trinetra browser extension and desktop app: real-time
deepfake detection on video-call face crops, contact/identity verification, scam-likelihood
scoring, and offline image/video analysis.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`torch` is pinned in `requirements.txt` but a plain `pip install -r requirements.txt` will
pull whatever default wheel PyPI serves (often CPU-only). For GPU inference, install torch
separately first with the index matching your CUDA version, e.g.:

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

## Run

```bash
uvicorn main:app --reload
```

The API is served at `http://localhost:8000` by default.

## API

See the docstring at the top of `main.py` for the full endpoint contract (request/response
shapes). Summary:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score a single pre-cropped face image; optional `contact_id` adds identity match + scam likelihood |
| `POST /analyze-image` | Detect and score every face in an uploaded photo |
| `POST /analyze-video` | Sample and score frames from an uploaded video |
| `POST /contacts`, `GET /contacts` | Enroll / list trusted contacts (reference face embeddings) |
| `POST /families`, `.../join`, `.../members`, `.../approve`, `.../reject` | Family Circle passcode-based membership flow |
| `GET /history` | Recent predictions log (backs the dashboard) |
| `GET /health` | Liveness check |
| `GET /model-info` | Which model checkpoint/device is loaded |
| `GET /dashboard`, `/analyze-ui`, `/contacts-ui`, `/family-ui` | Built-in HTML UIs (see `static/`) |

## Project layout

- `main.py` — FastAPI app and route handlers
- `predictor.py` — deepfake classifier (DeepfakeBench UCF/SPSL/Xception ensemble)
- `preprocessing.py` — face-crop quality gate (blur/sharpness, size)
- `face_detection.py`, `face_embedding.py` — face detection and identity embeddings (InsightFace)
- `image_pipeline.py`, `video_pipeline.py` — offline image/video analysis pipelines
- `scam_score.py` — scam-likelihood heuristics
- `contacts_db.py`, `family_db.py`, `history_db.py` — SQLite-backed storage
- `config.py` — centralized, environment-overridable configuration
- `scripts/` — one-off dev tools (sharpness calibration, benchmarking, test video generation)
- `tests/` — pytest suite

## Configuration

All tunables (upload limits, quality thresholds, classification thresholds, video sampling)
live in `config.py` and can be overridden via environment variables — see that file for the
full list and the reasoning behind each default.

## Tests

```bash
pytest
```

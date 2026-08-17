# Trinetra

Real-time deepfake detection for video calls — Google Meet, Zoom, Microsoft Teams, Discord,
and WhatsApp (web and desktop). Trinetra watches faces during a call, flags likely deepfakes,
and cross-checks speakers against known contacts to catch impersonation and scam attempts.

## Components

| Path | What it is |
|---|---|
| `backend/` | FastAPI service — deepfake classification, face detection/embedding, contact & family-circle identity checks, scam-likelihood scoring, history |
| `extension/` | Manifest V3 browser extension — captures call video in-page, sends face crops to the backend, overlays a REAL/FAKE/UNCERTAIN badge |
| `desktop/` | PySide6 app for native clients a browser extension can't reach (WhatsApp Desktop, Zoom/Teams desktop apps) — captures a chosen window via the Windows Graphics Capture API instead |
| `frontend/` | React + Vite dashboard — upload images/video for offline analysis, manage contacts and family circle, view detection history |

All three clients (extension, desktop, frontend) talk to the same backend at
`http://127.0.0.1:8000`.

## Getting started

Start the backend first — everything else depends on it.

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then, depending on what you want to run:

**Browser extension** — load `extension/` as an unpacked extension
(`chrome://extensions` → Developer mode → Load unpacked).

**Desktop app**

```bash
cd desktop
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Dashboard**

```bash
cd frontend
npm install
npm run dev
```

See each component's own README/docstrings for details (`backend/README.md`, docstring at
the top of `desktop/main.py`).

## Tests

```bash
cd backend
pytest
```

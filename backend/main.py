"""
FastAPI backend for the Trinetra extension and desktop app.

Endpoint contract (must stay in sync with extension/api.js and
desktop/api_client.py):
    POST /predict
        multipart/form-data:
          file       -> JPEG image of one cropped face (required)
          platform   -> string identifying the source, e.g. a hostname or
                        "desktop:<window title>" (optional, for the history log)
          contact_id -> int, an enrolled contact's id (optional). When
                        given, the live face is compared against that
                        contact's reference embedding and the response
                        includes an identity_match block + a scam_likelihood
                        score.
    Response (JSON):
        {
          "fake_probability": float in [0, 1] | null,  -- null iff quality_issue is set
          "classification": "real" | "fake" | "uncertain" | "insufficient_quality",
          "quality_issue": null | str,  -- why the crop was too low quality to run
          "identity_match": null | { "contact_name": str, "similarity": float, "matched": bool },
          "scam_likelihood": float in [0, 1],
          "scam_breakdown": { ... },
          "processing_time_ms": float,
          "model_version": null | str,
          "request_id": str
        }
        File uploads are limited to config.MAX_IMAGE_UPLOAD_BYTES and must be
        image/jpeg, image/png, or image/webp -- see config.py.

    POST /contacts
        multipart/form-data: name (str), file (reference photo)
        Response: { "id": int, "name": str }
    GET /contacts
        List enrolled contacts (id, name, created_at).

    Family Circles (family_db.py) -- see the module docstring there for the
    (deliberately simple, passcode-based) auth model:
    POST /families
        multipart/form-data: name (str), passcode (str) -> { id, name }
    POST /families/{family_id}/join
        multipart/form-data: name (str), passcode (str),
          photos -> one or more reference face images (repeat the field for each)
        Creates a PENDING member request. Response: { id, name, status: "pending" }
    GET /families/{family_id}/members?passcode=...
        Head-of-family view: list pending/approved/rejected members.
    POST /families/{family_id}/members/{member_id}/approve  (form: passcode)
    POST /families/{family_id}/members/{member_id}/reject   (form: passcode)
    GET /family-ui
        Website for all of the above.

    POST /analyze-image
        multipart/form-data: file -> any photo (not pre-cropped; every face
        in it is detected server-side, unlike /predict).
        Response: { "faces": [...], "overall_classification", "overall_confidence",
                    "processing_time_ms", "model_version", "request_id" }
        See image_pipeline.py.
    POST /analyze-video
        multipart/form-data: file -> video/mp4|webm|quicktime|x-matroska,
        up to config.MAX_VIDEO_UPLOAD_BYTES / MAX_VIDEO_DURATION_SECONDS.
        Samples frames evenly (config.VIDEO_SAMPLE_FPS), scores the largest
        face per sampled frame, aggregates with median + suspicious-segment
        detection. See video_pipeline.py.
    GET /analyze-ui
        Upload dashboard: drag-and-drop image/video analysis.

    GET /history?limit=50
        Recent predictions (newest first), each with a small thumbnail --
        backs the dashboard page.
    GET /dashboard
        Simple HTML page rendering /history.
    GET /contacts-ui
        Simple HTML page to enroll contacts.
    GET /health
        Liveness check: { "status": "ok" }.
    GET /model-info
        Which checkpoint/device is actually loaded -- see predictor.get_model_info.

predictor.py is the only file that should need to change once a trained
model is available -- see the comment there. Shared config (upload limits,
thresholds) lives in config.py.
"""
import base64
import io
import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

import contacts_db
import family_db
import history_db
from config import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    ALLOWED_VIDEO_CONTENT_TYPES,
    IDENTITY_MATCH_THRESHOLD,
    MAX_IMAGE_UPLOAD_BYTES,
    MAX_VIDEO_UPLOAD_BYTES,
    THUMBNAIL_SIZE,
    classify_fake_probability,
)
from errors import ModelLoadError
from face_embedding import average_embeddings, cosine_similarity, get_face_embedding
from image_pipeline import analyze_image
from predictor import get_model_info, predict_fake_probability
from preprocessing import assess_face_crop_quality, sharpness_variance
from scam_score import compute_scam_score
from video_pipeline import VideoDecodeError, VideoTooLongError, analyze_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("meet_deepfake_detector")

app = FastAPI(title="Meet Deepfake Detector Backend")

# Extension content scripts call this API from whichever call-platform
# origin they're injected into -- keep this in sync with manifest.json's
# content_scripts "matches" list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://meet.google.com",
        "https://zoom.us",
        "https://teams.microsoft.com",
        "https://teams.live.com",
        "https://discord.com",
        "https://web.whatsapp.com",
        "http://localhost:5173",  # frontend/ Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"https://.*\.zoom\.us",  # Zoom web client uses per-account subdomains
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

history_db.init_db()
contacts_db.init_db()
family_db.init_db()


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Structured request logging: request id + path + status + latency.
    Never logs media bytes/content -- only metadata."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        logger.exception("request_id=%s path=%s failed after %sms", request_id, request.url.path, elapsed_ms)
        raise
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s path=%s status=%s elapsed_ms=%s",
        request_id,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Last-resort catch: never let an uncaught error (bad CUDA state, a
    # dependency raising something we didn't anticipate) leak a stack trace
    # or internal path to the client -- log it fully server-side instead.
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("request_id=%s unhandled error: %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


async def _read_validated_upload(
    file: UploadFile, *, allowed_content_types: set[str], max_bytes: int, kind: str
) -> bytes:
    """Shared upload guard for every endpoint that accepts a file: rejects
    unrecognized MIME types and oversized bodies before they reach decoding
    (PIL/pydub) or the model. Never trusts the client-supplied filename."""
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported {kind} content type: {file.content_type!r}",
        )

    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{kind} upload exceeds the {max_bytes // (1024 * 1024)}MB limit",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty {kind} upload")
    return data


def _decode_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not decode image file") from exc


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    platform: str | None = Form(default=None),
    contact_id: int | None = Form(default=None),
    family_member_id: int | None = Form(default=None),
):
    start = time.monotonic()
    image_bytes = await _read_validated_upload(
        file,
        allowed_content_types=ALLOWED_IMAGE_CONTENT_TYPES,
        max_bytes=MAX_IMAGE_UPLOAD_BYTES,
        kind="image",
    )
    image = _decode_image(image_bytes)

    quality_issue = assess_face_crop_quality(image)
    logger.info(
        "request_id=%s /predict crop=%s sharpness=%.1f%s",
        request.state.request_id, image.size, sharpness_variance(image),
        f" -- REJECTED: {quality_issue}" if quality_issue else "",
    )
    if quality_issue is not None:
        fake_probability = None
    else:
        try:
            # CPU/GPU-bound model call -- must not run directly on the
            # asyncio event loop, or every other in-flight request (other
            # video tiles' /predict calls, /health, ...) stalls until this
            # one finishes.
            fake_probability = await run_in_threadpool(predict_fake_probability, image)
        except ModelLoadError as exc:
            logger.error("request_id=%s model unavailable: %s", request.state.request_id, exc)
            raise HTTPException(status_code=503, detail="Detection model unavailable") from exc

    classification = classify_fake_probability(fake_probability)

    identity_match = None
    identity_similarity = None
    identity_contact_name = None

    reference_name = None
    reference_embedding = None

    if family_member_id is not None:
        member = family_db.get_approved_member_embeddings(family_member_id)
        if member is not None and member["face_embedding"] is not None:
            reference_name = member["name"]
            reference_embedding = member["face_embedding"]
    elif contact_id is not None:
        contact = contacts_db.get_contact_embedding(contact_id)
        if contact is not None:
            reference_name, reference_embedding = contact

    if reference_embedding is not None:
        live_embedding = await run_in_threadpool(get_face_embedding, image)
        if live_embedding is not None:
            identity_similarity = cosine_similarity(live_embedding, reference_embedding)
            identity_contact_name = reference_name
            identity_match = {
                "contact_name": reference_name,
                "similarity": round(identity_similarity, 3),
                "matched": identity_similarity >= IDENTITY_MATCH_THRESHOLD,
            }

    scam_result = compute_scam_score(
        face_fake_probability=fake_probability, face_identity_similarity=identity_similarity
    )

    thumbnail = image.copy()
    thumbnail.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=70)
    thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    if fake_probability is not None:
        # history.db's fake_probability column is NOT NULL, and the
        # dashboard's classify() doesn't have an insufficient-quality bucket
        # -- skip logging rather than mislabeling a quality-rejected crop as
        # "uncertain". The API response below still reports it accurately.
        history_db.insert_prediction(
            platform,
            fake_probability,
            thumbnail_b64,
            identity_contact_name=identity_contact_name,
            identity_similarity=identity_similarity,
            scam_likelihood=scam_result["scam_likelihood"],
        )

    processing_time_ms = round((time.monotonic() - start) * 1000, 1)
    return {
        "fake_probability": fake_probability,
        "classification": classification,
        "quality_issue": quality_issue,
        "identity_match": identity_match,
        "scam_likelihood": scam_result["scam_likelihood"],
        "scam_breakdown": scam_result["breakdown"],
        "processing_time_ms": processing_time_ms,
        "model_version": get_model_info()["model_version"] if fake_probability is not None else None,
        "request_id": request.state.request_id,
    }


@app.post("/analyze-image")
async def analyze_image_route(request: Request, file: UploadFile = File(...)):
    """
    Dashboard image analysis: unlike /predict (one pre-cropped face from a
    live call tile), this takes an arbitrary uploaded photo, detects every
    face in it server-side, and scores each independently.

    Response:
    {
      "faces": [
        {
          "bbox": {"x","y","width","height"} normalized 0->1,
          "detection_confidence": float,
          "fake_probability": float | null,
          "classification": "real"|"fake"|"uncertain"|"insufficient_quality",
          "quality_issue": null | str
        }, ...
      ],
      "overall_classification": "real"|"fake"|"uncertain"|"insufficient_quality"|"no_face_detected",
      "overall_confidence": float | null,
      "processing_time_ms": float,
      "model_version": str,
      "request_id": str
    }
    """
    start = time.monotonic()
    image_bytes = await _read_validated_upload(
        file,
        allowed_content_types=ALLOWED_IMAGE_CONTENT_TYPES,
        max_bytes=MAX_IMAGE_UPLOAD_BYTES,
        kind="image",
    )
    image = _decode_image(image_bytes)

    try:
        result = await run_in_threadpool(analyze_image, image)
    except ModelLoadError as exc:
        logger.error("request_id=%s model unavailable: %s", request.state.request_id, exc)
        raise HTTPException(status_code=503, detail="Detection model unavailable") from exc

    return {
        **result,
        "processing_time_ms": round((time.monotonic() - start) * 1000, 1),
        "model_version": get_model_info()["model_version"],
        "request_id": request.state.request_id,
    }


@app.post("/analyze-video")
async def analyze_video_route(request: Request, file: UploadFile = File(...)):
    """
    Dashboard video analysis: samples frames evenly across the video
    (config.VIDEO_SAMPLE_FPS, capped at MAX_VIDEO_SAMPLE_FRAMES), scores the
    largest face per sampled frame, and aggregates with outlier rejection
    (median) plus minimum-run-length suspicious-segment detection.

    Response:
    {
      "metadata": {"duration_seconds","fps","width","height","total_frames","frames_analyzed"},
      "frames": [{"timestamp","fake_probability","classification"}, ...],
      "suspicious_segments": [{"start_timestamp","end_timestamp","peak_fake_probability","frame_count"}, ...],
      "overall_classification": "real"|"fake"|"uncertain"|"insufficient_quality"|"no_face_detected",
      "overall_confidence": float | null,
      "processing_time_ms": float,
      "model_version": str,
      "request_id": str
    }
    """
    start = time.monotonic()
    if file.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail=f"Unsupported video content type: {file.content_type!r}"
        )

    video_bytes = await file.read(MAX_VIDEO_UPLOAD_BYTES + 1)
    if len(video_bytes) > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"video upload exceeds the {MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)}MB limit",
        )
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video upload")

    # cv2.VideoCapture needs a real file path (no in-memory decode API) --
    # written under a throwaway temp dir, cleaned up unconditionally, and
    # never named from the client-supplied filename (extension is looked up
    # from the validated content-type only).
    suffix = ALLOWED_VIDEO_CONTENT_TYPES[file.content_type]
    with tempfile.TemporaryDirectory(prefix="deepfake_upload_") as tmp_dir:
        video_path = Path(tmp_dir) / f"upload{suffix}"
        video_path.write_bytes(video_bytes)

        try:
            result = await run_in_threadpool(analyze_video, video_path)
        except VideoTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except VideoDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelLoadError as exc:
            logger.error("request_id=%s model unavailable: %s", request.state.request_id, exc)
            raise HTTPException(status_code=503, detail="Detection model unavailable") from exc

    return {
        **result,
        "processing_time_ms": round((time.monotonic() - start) * 1000, 1),
        "model_version": get_model_info()["model_version"],
        "request_id": request.state.request_id,
    }


@app.get("/analyze-ui")
async def analyze_ui():
    return FileResponse("static/analyze.html")


@app.post("/contacts")
async def enroll_contact(name: str = Form(...), file: UploadFile = File(...)):
    if not name.strip():
        raise HTTPException(status_code=400, detail="Name is required")

    image_bytes = await _read_validated_upload(
        file,
        allowed_content_types=ALLOWED_IMAGE_CONTENT_TYPES,
        max_bytes=MAX_IMAGE_UPLOAD_BYTES,
        kind="image",
    )
    image = _decode_image(image_bytes)

    embedding = await run_in_threadpool(get_face_embedding, image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in reference photo")

    contact_id = contacts_db.insert_contact(name.strip(), embedding)
    return {"id": contact_id, "name": name.strip()}


@app.get("/contacts")
async def list_contacts():
    return contacts_db.list_contacts()


@app.get("/contacts-ui")
async def contacts_ui():
    return FileResponse("static/contacts.html")


@app.post("/families")
async def create_family(name: str = Form(...), passcode: str = Form(...)):
    family_id = family_db.create_family(name, passcode)
    return {"id": family_id, "name": name}


@app.post("/families/{family_id}/join")
async def join_family(
    family_id: int,
    name: str = Form(...),
    passcode: str = Form(...),
    photos: list[UploadFile] = File(...),
):
    if not family_db.verify_family_passcode(family_id, passcode):
        raise HTTPException(status_code=403, detail="Wrong family ID or passcode")

    embeddings = []
    for photo in photos:
        image_bytes = await _read_validated_upload(
            photo,
            allowed_content_types=ALLOWED_IMAGE_CONTENT_TYPES,
            max_bytes=MAX_IMAGE_UPLOAD_BYTES,
            kind="image",
        )
        image = _decode_image(image_bytes)
        embedding = await run_in_threadpool(get_face_embedding, image)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        raise HTTPException(status_code=400, detail="No face detected in any of the submitted photos")

    face_embedding = average_embeddings(embeddings)

    member_id = family_db.create_member_request(family_id, name, face_embedding)
    return {"id": member_id, "name": name, "status": "pending"}


@app.get("/families/{family_id}/members")
async def get_family_members(family_id: int, passcode: str):
    if not family_db.verify_family_passcode(family_id, passcode):
        raise HTTPException(status_code=403, detail="Wrong family ID or passcode")
    return family_db.list_members(family_id)


@app.post("/families/{family_id}/members/{member_id}/approve")
async def approve_member(family_id: int, member_id: int, passcode: str = Form(...)):
    if not family_db.verify_family_passcode(family_id, passcode):
        raise HTTPException(status_code=403, detail="Wrong family ID or passcode")
    family_db.set_member_status(member_id, "approved")
    return {"id": member_id, "status": "approved"}


@app.post("/families/{family_id}/members/{member_id}/reject")
async def reject_member(family_id: int, member_id: int, passcode: str = Form(...)):
    if not family_db.verify_family_passcode(family_id, passcode):
        raise HTTPException(status_code=403, detail="Wrong family ID or passcode")
    family_db.set_member_status(member_id, "rejected")
    return {"id": member_id, "status": "rejected"}


@app.get("/family-ui")
async def family_ui():
    return FileResponse("static/family.html")


@app.get("/history")
async def history(limit: int = 50):
    return history_db.get_recent(limit)


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model-info")
async def model_info():
    """Exposes which checkpoint is actually loaded and its device, so a
    client/operator can confirm the running model matches expectations
    rather than assuming it silently."""
    return await run_in_threadpool(get_model_info)

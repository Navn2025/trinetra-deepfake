"""
Voice anti-spoofing service for the Trinetra extension and desktop app.

Deliberately separate from ../backend/ (the face-detection FastAPI service,
port 8000): that service's venv can't host this one's dependency stack
without real conflicts -- fairseq, speechbrain, and the hand-patched
LocalStrategy/merge_with_parent/Windows-path-separator fixes documented in
xlsr_sls_detector.py and speech_verification.py all live in
voice-integrity/venv specifically because getting them working required
isolating this environment. Run with:
    cd voice-integrity/src && ..\\venv\\Scripts\\python.exe -m uvicorn server:app --port 8001
(must run from inside src/, same as the other scripts here -- server.py
imports sibling modules like audio_utils with bare `import audio_utils`,
which only resolves if src/ itself is on sys.path.)

Endpoint contract:
    POST /voice/enroll
        multipart/form-data: name (str), file (reference audio -- a clean
        sample of the person's genuine voice, the longer/cleaner the
        better; speechbrain resamples internally, no fixed format needed).
        Saves it under data/enrolled/<slug(name)>.<ext>, overwriting any
        previous enrollment for that name. Response: {"name": slug, "saved": true}.
    GET /voice/enrolled
        Response: {"names": [str, ...]} -- every enrolled name (slugified).
    GET /voice/enrolled/{name}/audio
        Streams the enrolled WAV back (audio/wav) so the caller can listen
        to it and confirm the right person/sample actually got enrolled,
        rather than trusting the name alone. 404 if not enrolled.
    DELETE /voice/enrolled/{name}
        Removes an enrollment. Response: {"name": slug, "deleted": true},
        or 404 if that name was never enrolled.
    POST /voice/check
        multipart/form-data:
          file           -> any audio file (any sample rate/format soundfile
                             can read; auto-resampled to 16kHz mono, see
                             audio_utils.py)
          enhance        -> "true"/"1" (optional, default false) -- see
                             audio_enhance.py. When set, the clip is denoised
                             + run through a neural speech enhancer *before*
                             scoring, and the top-level fields below reflect
                             the enhanced clip; the pre-enhancement scores
                             are still included under "enhancement.raw" for
                             comparison -- see audio_enhance.py's docstring
                             for why this is a second opinion, not a strict
                             improvement.
          reference_name -> optional name of a prior /voice/enroll upload.
                             When given, also runs speaker-identity
                             verification (pipeline.py's check #1) against
                             that enrollment and returns it under
                             "identity". 404 if that name isn't enrolled.
        Response:
        {
          "gustking": {"real": float, "fake": float},          -- whole clip
          "xlsr_sls": {"bonafide": float, "spoof": float},      -- whole clip
          "verdict": "FAKE/SPOOFED" | "LIKELY GENUINE",
          "identity": null | {
            "reference_name": str, "similarity": float, "same_speaker": bool
          },
          "elapsed_s": float,
          "segments": [
            {"start": float, "end": float,
             "gustking": {...}, "xlsr_sls": {...}, "flagged": bool}, ...
          ],
          "most_suspicious_segment": <one of the above> | null,
          "enhancement": null | {
            "applied": true,
            "denoiser": str, "enhancer": str,
            "raw": {"gustking": {...}, "xlsr_sls": {...}, "verdict": str}
          }
        }
        segments/most_suspicious_segment localize *where in the clip* the
        spoofing signal is strongest, by re-running both detectors over a
        sliding window instead of just the whole clip -- see
        _score_segments() below. "identity" answers "is this the enrolled
        person's voice", which is independent of "verdict" answering "is
        this voice synthetic" -- a convincing clone of the reference voice
        will legitimately show same_speaker=true (see pipeline.py's
        docstring for why identity is reported, not used to override the
        spoof verdict).
    GET /health
        Liveness check: { "status": "ok" }.

Models are loaded once at startup, not per-request -- XLS-R+SLS's load cost
(a real 300M-param transformer) would otherwise be paid on every call.
"""
import asyncio
import logging
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

import audio_enhance
import audio_utils
import speech_verification
import synthetic_detector
import xlsr_sls_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("voice_integrity")

SPOOF_THRESHOLD = 0.5

# -- Voice enrollment -----------------------------------------------------
# Reference clips for speaker-identity matching (pipeline.py's check #1,
# now reachable from the API instead of CLI-only). One raw audio file per
# enrolled name; verify() re-derives the speaker embedding from the file on
# every check rather than caching an embedding, since these enrollments are
# few and infrequent enough that re-embedding costs nothing worth caching
# for -- see speech_verification.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
ENROLLED_DIR = REPO_ROOT / "data" / "enrolled"

# -- Segment localization -----------------------------------------------
# A whole-clip verdict doesn't tell the caller *where* the spoofing signal
# actually is (a real intro dubbed with 2s of cloned audio scores as
# "somewhat spoofed" on average, hiding exactly the part that matters).
# Re-run both detectors over a sliding window instead: 4s roughly matches
# xlsr_sls_detector's own CUT_LEN (~4.04s, what the SLS head was trained
# on -- shorter windows would be padded/tiled away from their natural
# input shape and read noisier), with 50% overlap (2s stride) so a short
# spliced segment can't land entirely on a window boundary and get diluted
# into two windows that each only partly contain it.
SEGMENT_WINDOW_SECONDS = 4.0
SEGMENT_STRIDE_SECONDS = 2.0
# Bounds total cost for a long upload -- each window pays a full XLS-R 300M
# forward pass, and on this deployment's GPU that's ~0.7-0.9s/window with no
# way to make it cheaper via batching or concurrency (see
# _score_segments_sync's docstring -- both were tried and measured *worse*
# than plain sequential calls). So this constant is the only real lever on
# segment-scoring latency: at 12 it was adding ~10-13s to every check just
# for windows on top of the ~1s whole-clip verdict. 4 evenly-spread windows
# (start/early/late/end) still localizes roughly where in the clip the
# signal is strongest, at a third of the cost; longer clips fall back to
# fewer, evenly-spread windows rather than growing unboundedly.
MAX_SEGMENTS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The two anti-spoofing detectors are eager-loaded since every request
    # needs them. audio_enhance's MetricGAN+ enhancer is deliberately NOT
    # loaded here -- it's opt-in per request (enhance=true) and most
    # requests won't use it, so its first-use load/download cost (a
    # HuggingFace fetch on a cold cache) is paid lazily on the first
    # enhanced request rather than slowing down every server startup.
    logger.info("loading voice models (Gustking, XLS-R+SLS, speaker verification)...")
    start = time.monotonic()
    ENROLLED_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_detector.load_model()
    xlsr_sls_detector.load_model()
    # ECAPA-TDNN is ~20MB, cheap enough to eager-load alongside the other
    # two rather than paying its load cost on whichever request happens to
    # be the first one that passes reference_name.
    speech_verification.load_verifier()
    logger.info("voice models ready in %.1fs", time.monotonic() - start)
    yield


app = FastAPI(title="Trinetra Voice Integrity Service", lifespan=lifespan)

# Same origin set as backend/main.py -- the extension's content script calls
# this from whichever call-platform origin it's injected into, plus the
# desktop app and dashboard frontend calling from localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://meet.google.com",
        "https://zoom.us",
        "https://teams.microsoft.com",
        "https://teams.live.com",
        "https://discord.com",
        "https://web.whatsapp.com",
    ],
    # See backend/main.py's identical comment -- the frontend/ Vite dev
    # server's port shifts whenever 5173 is already taken, so this matches
    # any localhost/127.0.0.1 port rather than hardcoding one.
    allow_origin_regex=r"https://.*\.zoom\.us|http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _is_flagged(gustking_fake_prob: float, spoof_prob: float) -> bool:
    """Shared flagging rule for both the whole-clip verdict and each
    segment. Gustking-only: on this deployment's real call-audio (compressed
    WhatsApp/video-call recordings), xlsr_sls has repeatedly scored known-
    genuine clips at ~100% spoof with high confidence -- not a borderline
    miss but a confident wrong answer, which defeated every agreement/
    override combination tried (AND, AND-with-high-confidence-override; see
    prior revisions of this function). Gustking has been correct on every
    real-world sample tested so far. xlsr_sls's score is still computed and
    returned in the response for visibility/comparison, just not used to
    decide the verdict. spoof_prob is intentionally unused -- kept as a
    parameter so call sites and the segment loop don't need to change if a
    second opinion is reinstated later."""
    del spoof_prob
    return gustking_fake_prob >= SPOOF_THRESHOLD


def _segment_window_starts(duration_s: float) -> list[float]:
    """Evenly-spread window start times covering [0, duration_s], targeting
    SEGMENT_WINDOW_SECONDS/SEGMENT_STRIDE_SECONDS but capped at MAX_SEGMENTS
    (see that constant's comment)."""
    if duration_s <= SEGMENT_WINDOW_SECONDS:
        return [0.0]

    span = duration_s - SEGMENT_WINDOW_SECONDS
    target_count = min(MAX_SEGMENTS, int(span / SEGMENT_STRIDE_SECONDS) + 2)
    if target_count <= 1:
        return [0.0]
    step = span / (target_count - 1)
    return [round(i * step, 3) for i in range(target_count)]


def _score_segments_sync(windows: list[tuple[np.ndarray, float, float]], sample_rate: int) -> list[dict]:
    """Scores every window sequentially (not batched -- measured on this
    deployment's GPU: batching 12 windows into one forward pass was
    *slower* than 12 sequential calls for both models, 13.25s vs 10.12s for
    xlsr_sls and 4.33s vs 2.05s for gustking, presumably because this GPU
    has no spare parallel capacity and a bigger batched tensor just adds
    overhead. Concurrent threadpool scheduling (asyncio.gather over
    run_in_threadpool) was tried before that and also showed no real
    speedup, for the same reason: this is genuinely GPU-compute-bound work,
    not I/O or Python-overhead-bound. See MAX_SEGMENTS' comment for the
    actual lever that helps -- doing less of this work, not scheduling it
    differently.)"""
    arrays = [w for w, _, _ in windows]
    gustking_results = [synthetic_detector.predict_array(a, sample_rate) for a in arrays]
    sls_results = [xlsr_sls_detector.predict_array(a, sample_rate) for a in arrays]

    segments = []
    for (_, start, end), gustking_labels, sls_result in zip(windows, gustking_results, sls_results):
        gustking_fake_prob = gustking_labels.get("fake", max(gustking_labels.values()))
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "gustking": gustking_labels,
            "xlsr_sls": sls_result,
            "flagged": _is_flagged(gustking_fake_prob, sls_result["spoof"]),
        })
    return segments


async def _score_segments(audio_16k_path: Path) -> list[dict]:
    """Re-scores overlapping SEGMENT_WINDOW_SECONDS windows of the (already
    16kHz-mono) clip with both anti-spoofing detectors, so the caller can
    see which part of the clip the spoofing signal is strongest in rather
    than only a single whole-clip average."""
    samples, sample_rate = sf.read(str(audio_16k_path))
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration_s = len(samples) / sample_rate

    windows = []
    for start in _segment_window_starts(duration_s):
        end = min(start + SEGMENT_WINDOW_SECONDS, duration_s)
        window = samples[int(start * sample_rate):int(end * sample_rate)]
        if window.size > 0:
            windows.append((window.astype(np.float32), start, end))

    return await run_in_threadpool(_score_segments_sync, windows, sample_rate)


def _most_suspicious(segments: list[dict]) -> dict | None:
    """The window with the strongest spoofing signal, flagged or not --
    like the video pipeline's most_suspicious_frame, this is "what the
    model reacted to most," not a certified verdict on its own."""
    if not segments:
        return None
    return max(segments, key=lambda s: max(s["gustking"].get("fake", 0.0), s["xlsr_sls"]["spoof"]))


async def _score_clip(audio_path: Path) -> dict:
    """Whole-clip gustking + xlsr_sls scores, verdict, and windowed
    segments for one already-16kHz-mono file -- the common scoring step
    run once for the raw upload and, when enhance=true, a second time for
    the enhanced copy. The whole-clip pass and the windowed-segment pass are
    independent, so they run concurrently rather than one after the other."""
    (gustking_labels, sls_result), segments = await asyncio.gather(
        asyncio.gather(
            run_in_threadpool(synthetic_detector.predict, audio_path),
            run_in_threadpool(xlsr_sls_detector.predict, audio_path),
        ),
        _score_segments(audio_path),
    )
    gustking_fake_prob = gustking_labels.get("fake", max(gustking_labels.values()))
    return {
        "gustking": gustking_labels,
        "xlsr_sls": sls_result,
        "verdict": "FAKE/SPOOFED" if _is_flagged(gustking_fake_prob, sls_result["spoof"]) else "LIKELY GENUINE",
        "segments": segments,
        "most_suspicious_segment": _most_suspicious(segments),
    }


def _safe_enrolled_name(name: str) -> str:
    """Slugifies an enrollment name into a safe filename stem -- never
    derived-into a path from unsanitized user input, since this ends up in
    a filesystem path under ENROLLED_DIR."""
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    if not slug:
        raise HTTPException(status_code=400, detail="Name must contain at least one alphanumeric character")
    return slug


def _find_enrolled(name: str) -> Path | None:
    slug = _safe_enrolled_name(name)
    matches = list(ENROLLED_DIR.glob(f"{slug}.*"))
    return matches[0] if matches else None


@app.post("/voice/enroll")
async def voice_enroll(name: str = Form(...), file: UploadFile = File(...)):
    """Saves a reference voice sample under `name` for later speaker-identity
    matching in /voice/check. Re-enrolling the same name overwrites the
    previous sample. Always normalized to a canonical {slug}.wav (16kHz
    mono) regardless of what was uploaded -- including extracting audio
    from a video file via ensure_readable_audio(), the same fallback
    /voice/check uses, so a raw video upload here doesn't get saved
    unreadable and only fail the first time it's used as a reference."""
    slug = _safe_enrolled_name(name)
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="voice_enroll_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        upload_path = tmp_dir / f"upload{suffix}"
        upload_path.write_bytes(audio_bytes)
        try:
            readable_path = await run_in_threadpool(audio_utils.ensure_readable_audio, upload_path, tmp_dir)
            normalized_path = await run_in_threadpool(audio_utils.resample_to_16k_mono, readable_path, tmp_dir)
        except Exception as exc:
            logger.exception("enrollment failed: %s", exc)
            raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}") from exc

        for existing in ENROLLED_DIR.glob(f"{slug}.*"):
            existing.unlink()
        dest = ENROLLED_DIR / f"{slug}.wav"
        dest.write_bytes(normalized_path.read_bytes())

    logger.info("enrolled reference voice %r -> %s", name, dest.name)
    return {"name": slug, "saved": True}


@app.get("/voice/enrolled")
async def voice_enrolled_list():
    names = sorted({p.stem for p in ENROLLED_DIR.glob("*") if p.is_file()})
    return {"names": names}


@app.get("/voice/enrolled/{name}/audio")
async def voice_enrolled_audio(name: str):
    """Streams the enrolled reference sample back so the caller can verify
    by ear that the right person actually got enrolled -- always a 16kHz
    mono WAV, since /voice/enroll normalizes every upload to that."""
    path = _find_enrolled(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No enrolled voice named {name!r}")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.delete("/voice/enrolled/{name}")
async def voice_enrolled_delete(name: str):
    slug = _safe_enrolled_name(name)
    matches = list(ENROLLED_DIR.glob(f"{slug}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No enrolled voice named {slug!r}")
    for match in matches:
        match.unlink()
    return {"name": slug, "deleted": True}


@app.post("/voice/check")
async def voice_check(
    file: UploadFile = File(...),
    enhance: bool = Form(default=False),
    reference_name: str | None = Form(default=None),
):
    start = time.monotonic()
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload")

    reference_path = None
    if reference_name:
        reference_path = _find_enrolled(reference_name)
        if reference_path is None:
            raise HTTPException(status_code=404, detail=f"No enrolled voice named {reference_name!r}")

    suffix = Path(file.filename or "clip.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory(prefix="voice_upload_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        upload_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
        upload_path.write_bytes(audio_bytes)

        try:
            # The frontend extracts audio client-side before uploading, but
            # other callers (desktop app, direct API use) may not -- fall
            # back to ffmpeg extraction here too rather than failing on a
            # raw video upload. Resampled copy (if needed) lives in the same
            # per-request temp dir, not the CLI pipeline's persistent
            # data/.cache/ -- every upload here is a one-shot, unnamed clip,
            # so there's nothing to usefully cache and doing so would just
            # grow unbounded on a long-running service polled every 30s.
            readable_path = await run_in_threadpool(audio_utils.ensure_readable_audio, upload_path, tmp_dir)
            audio_16k = await run_in_threadpool(audio_utils.resample_to_16k_mono, readable_path, tmp_dir)

            async def _identity_task() -> dict | None:
                if reference_path is None:
                    return None
                # Identity only, run against the raw (un-enhanced) upload --
                # denoising is meant to help anti-spoofing scoring, not
                # identity matching, and shouldn't change whose voice this
                # is. Same speaker != genuine: a convincing clone of the
                # reference voice will legitimately pass this (see
                # pipeline.py's docstring) -- it's reported alongside the
                # spoof verdict, not as a substitute for it.
                # speechbrain's verify_files() mis-parses a Windows
                # backslash path with a drive letter (e.g. "C:\Users\..."
                # loses the backslash right after "C:", becoming
                # "C:Users\..." and failing to open) -- .as_posix() sidesteps
                # it entirely, same as the other hand-patched Windows-path
                # fixes noted in this module's docstring.
                similarity, same_speaker = await run_in_threadpool(
                    speech_verification.verify, reference_path.as_posix(), audio_16k.as_posix()
                )
                return {
                    "reference_name": reference_path.stem,
                    "similarity": similarity,
                    "same_speaker": same_speaker,
                }

            async def _enhanced_task() -> dict | None:
                if not enhance:
                    return None
                # See audio_enhance.py's module docstring: enhancement is a
                # second opinion, not a strict improvement, so the raw
                # scores are always kept alongside the enhanced ones rather
                # than discarded.
                enhanced_path = await run_in_threadpool(audio_enhance.enhance_audio, audio_16k, tmp_dir)
                return await _score_clip(enhanced_path)

            # raw scoring, identity matching, and enhanced-clip scoring are
            # all independent of each other -- run them concurrently rather
            # than stacking their wall-clock costs.
            raw_result, identity, enhanced_result = await asyncio.gather(
                _score_clip(audio_16k), _identity_task(), _enhanced_task()
            )

            enhancement = None
            result = raw_result
            if enhance:
                result = enhanced_result
                enhancement = {
                    "applied": True,
                    "denoiser": "noisereduce (spectral gating, non-stationary)",
                    "enhancer": "speechbrain/metricgan-plus-voicebank",
                    "raw": {
                        "gustking": raw_result["gustking"],
                        "xlsr_sls": raw_result["xlsr_sls"],
                        "verdict": raw_result["verdict"],
                    },
                }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("voice check failed: %s", exc)
            raise HTTPException(status_code=422, detail=f"Could not analyze audio: {exc}") from exc

    return {
        **result,
        "identity": identity,
        "elapsed_s": round(time.monotonic() - start, 2),
        "enhancement": enhancement,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

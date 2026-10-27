"""
main.py — Voice Lock
FastAPI microservice: speaker verification for the shop billing pipeline.

Endpoints
---------
POST /enroll   — register owner voice (call 3-5 times with different clips)
POST /verify   — check if audio clip belongs to the enrolled owner
GET  /health   — boot check used by Node backend (checkVoiceService)

Run
---
venv\\Scripts\\uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

# Windows terminal encoding fix
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONUTF8"] = "1"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import speaker
import store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice_lock")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

THRESHOLD = 0.30          # cosine similarity cutoff — tune in Phase 5
DEFAULT_SPEAKER = "shop_1_owner"

# ---------------------------------------------------------------------------
# Lifespan — load model ONCE at startup, not per request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ECAPA model before the server starts accepting requests."""
    logger.info("=== Voice Lock service starting up ===")
    logger.info("Loading ECAPA-TDNN model (this takes ~5s on warm disk)...")
    speaker.load_model()
    logger.info("Model ready. Serving on port 8000.")
    yield
    logger.info("=== Voice Lock service shutting down ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice Lock — Speaker Verification Service",
    description=(
        "Verifies that an audio clip belongs to the enrolled shop owner. "
        "Only verified audio is forwarded to Sarvam STT for transcription."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(upload: UploadFile) -> str:
    """Save an uploaded file to a temp path and return that path."""
    suffix = Path(upload.filename).suffix if upload.filename else ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with tmp:
        shutil.copyfileobj(upload.file, tmp)
    return tmp.name


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Boot check for Node backend")
def health():
    """
    Returns service status and model name.
    Called by Node backend at startup (checkVoiceService).
    Must respond quickly — model is already loaded.
    """
    enrolled = store.is_enrolled(DEFAULT_SPEAKER)
    return {
        "status": "ok",
        "model": "ecapa-voxceleb",
        "enrolled": enrolled,
        "threshold": THRESHOLD,
    }


@app.post("/enroll", summary="Register owner voice embedding")
async def enroll(
    audio: UploadFile = File(..., description="Audio clip (.m4a or .wav), min 1.5s"),
    speaker_id: str = Form(DEFAULT_SPEAKER, description="Speaker identifier"),
):
    """
    Add a new voice sample for the owner.
    Call 3-5 times with different clips for a stable mean embedding.

    Response shape (contract with Node backend):
    {
      "speaker_id": "shop_1_owner",
      "embedding_stored": true,
      "samples": 3,
      "similarity_mean": 0.82
    }
    """
    tmp_path = _save_upload(audio)

    try:
        vec = speaker.embed(tmp_path)
    except speaker.ClipTooShortError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Embed failed during /enroll: %s", e)
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    mean_vec, samples = store.update_mean(speaker_id, vec)
    similarity_to_mean = speaker.compare(mean_vec, vec)

    logger.info(
        "/enroll speaker='%s' samples=%d similarity_to_mean=%.4f",
        speaker_id, len(samples), similarity_to_mean,
    )

    return {
        "speaker_id": speaker_id,
        "embedding_stored": True,
        "samples": len(samples),
        "similarity_mean": round(float(similarity_to_mean), 4),
    }


@app.post("/verify", summary="Verify if audio belongs to enrolled owner")
async def verify(
    audio: UploadFile = File(..., description="Audio clip (.m4a or .wav), min 1.5s"),
    speaker_id: str = Form(DEFAULT_SPEAKER, description="Speaker identifier"),
):
    """
    Compare audio clip against stored owner embedding.

    IMPORTANT: A low score is NOT an HTTP error.
    Only throw 4xx/5xx on real errors (bad audio, not enrolled, model missing).

    Response shape (contract with Node backend):
    {
      "speaker_id": "shop_1_owner",
      "score": 0.41,
      "is_speaker": true,
      "threshold": 0.30
    }
    """
    # Check enrollment before processing audio (fast fail)
    if not store.is_enrolled(speaker_id):
        raise HTTPException(
            status_code=404,
            detail=f"Speaker '{speaker_id}' is not enrolled. Call /enroll first.",
        )

    tmp_path = _save_upload(audio)

    try:
        vec = speaker.embed(tmp_path)
    except speaker.ClipTooShortError:
        # Not an error — return a definitive "not speaker" with reason
        logger.info("/verify speaker='%s' result=too_short", speaker_id)
        return JSONResponse({
            "speaker_id": speaker_id,
            "score": 0.0,
            "is_speaker": False,
            "threshold": THRESHOLD,
            "reason": "too_short",
        })
    except Exception as e:
        logger.error("Embed failed during /verify: %s", e)
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    mean_vec, _ = store.load_embedding(speaker_id)
    score = speaker.compare(mean_vec, vec)
    is_speaker = bool(score >= THRESHOLD)

    logger.info(
        "/verify speaker='%s' score=%.4f is_speaker=%s threshold=%.2f",
        speaker_id, score, is_speaker, THRESHOLD,
    )

    return {
        "speaker_id": speaker_id,
        "score": round(float(score), 4),
        "is_speaker": is_speaker,
        "threshold": THRESHOLD,
    }


@app.post("/embed", summary="Return raw embedding vector (debugging)")
async def embed_only(
    audio: UploadFile = File(...),
):
    """
    Returns the raw 192-dim embedding for an audio clip.
    Useful for debugging and threshold calibration (Phase 5).
    Not called by Node backend in production.
    """
    tmp_path = _save_upload(audio)
    try:
        vec = speaker.embed(tmp_path)
    except speaker.ClipTooShortError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"embedding": vec.tolist(), "shape": list(vec.shape)}

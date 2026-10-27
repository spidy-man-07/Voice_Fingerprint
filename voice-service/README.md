# 🔐 Voice Lock — Speaker Verification Service

A FastAPI microservice that verifies whether an audio clip belongs to the enrolled shop owner using **ECAPA-TDNN** voice embeddings. Only verified audio is forwarded to Sarvam STT for transcription — strangers, TV, and background noise are silently ignored.

---

## How It Works

```
Audio clip (.m4a / .wav)
        │
        ▼
   ffmpeg decode ──► 16 kHz mono PCM
        │
        ▼
   ECAPA-TDNN model ──► 192-dim voice embedding (fingerprint)
        │
        ▼
   Cosine similarity vs. stored owner embedding
        │
        ├── score ≥ 0.30 → ✅ Owner confirmed → forward to STT
        └── score < 0.30 → ❌ Rejected (stranger / noise)
```

**Key idea:** This is a **classification-threshold** problem, not speech recognition. The pretrained ECAPA-TDNN model (trained on 7,000+ speakers) produces a unique voice fingerprint. We compare fingerprints — we don't train anything.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.10+ | Tested on 3.14.0 |
| **ffmpeg** | Any recent | Must be on system `PATH` |
| **Disk space** | ~500 MB | Model (~200 MB) + PyTorch + dependencies |
| **OS** | Windows / Linux / macOS | Tested on Windows |

---

## Quick Start

### 1. Clone & Setup

```powershell
cd voice-service

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify ffmpeg

```powershell
ffmpeg -version
# Should print version info. If not found, install ffmpeg and add to PATH.
```

### 3. Run Smoke Test (Optional)

```powershell
python smoke_test.py
```

First run downloads the ECAPA-TDNN model (~200 MB) to `pretrained_models/ecapa/`. Subsequent runs load from disk in ~3s.

### 4. Start the Service

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The model loads at startup (~3–5s). Once you see `Model ready. Serving on port 8000.` the service is accepting requests.

**Docs UI:** Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive Swagger documentation.

---

## API Reference

### `GET /health` — Health Check

Returns service status. Used by the Node backend at boot.

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "ecapa-voxceleb",
  "enrolled": false,
  "threshold": 0.30
}
```

---

### `POST /enroll` — Register Owner Voice

Enroll the shop owner's voice. **Call 3–5 times** with different audio clips for a stable mean embedding.

```bash
# Enroll with clip 1
curl -X POST http://127.0.0.1:8000/enroll \
  -F "audio=@clip1.wav" \
  -F "speaker_id=shop_1_owner"

# Enroll with clip 2
curl -X POST http://127.0.0.1:8000/enroll \
  -F "audio=@clip2.wav" \
  -F "speaker_id=shop_1_owner"

# Enroll with clip 3
curl -X POST http://127.0.0.1:8000/enroll \
  -F "audio=@clip3.wav" \
  -F "speaker_id=shop_1_owner"
```

**PowerShell equivalent:**
```powershell
# PowerShell uses -Form for multipart uploads
Invoke-RestMethod -Uri "http://127.0.0.1:8000/enroll" -Method Post -Form @{
    audio    = Get-Item "clip1.wav"
    speaker_id = "shop_1_owner"
}
```

**Response:**
```json
{
  "speaker_id": "shop_1_owner",
  "embedding_stored": true,
  "samples": 3,
  "similarity_mean": 0.82
}
```

| Field | Description |
|---|---|
| `samples` | Total clips enrolled so far |
| `similarity_mean` | How similar this clip is to the running mean (higher = more consistent) |

**Tips for good enrollment:**
- Record at least **3 clips**, each **3–5 seconds** long
- Speak **different sentences** each time (variety helps)
- Record from the **same device** the app will use (phone mic, not laptop mic)
- Record in the **same environment** as production (the shop)

---

### `POST /verify` — Verify Speaker Identity

Check if an audio clip belongs to the enrolled owner. **A low score is NOT an HTTP error** — it's a valid "not the speaker" result.

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -F "audio=@test_clip.wav" \
  -F "speaker_id=shop_1_owner"
```

**PowerShell equivalent:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/verify" -Method Post -Form @{
    audio      = Get-Item "test_clip.wav"
    speaker_id = "shop_1_owner"
}
```

**Response (owner recognised):**
```json
{
  "speaker_id": "shop_1_owner",
  "score": 0.41,
  "is_speaker": true,
  "threshold": 0.30
}
```

**Response (stranger / noise):**
```json
{
  "speaker_id": "shop_1_owner",
  "score": 0.12,
  "is_speaker": false,
  "threshold": 0.30
}
```

**Response (clip too short):**
```json
{
  "speaker_id": "shop_1_owner",
  "score": 0.0,
  "is_speaker": false,
  "threshold": 0.30,
  "reason": "too_short"
}
```

| Field | Description |
|---|---|
| `score` | Cosine similarity (−1.0 to 1.0). Higher = more likely the owner |
| `is_speaker` | `true` if `score ≥ threshold` |
| `threshold` | Current cutoff (default 0.30, tuned in Phase 5) |
| `reason` | Present only when audio is < 1.5 s (`"too_short"`) |

---

### `POST /embed` — Raw Embedding (Debug)

Returns the raw 192-dimensional embedding vector for an audio clip. Useful for threshold calibration and debugging.

```bash
curl -X POST http://127.0.0.1:8000/embed -F "audio=@clip.wav"
```

**Response:**
```json
{
  "embedding": [0.0123, -0.0456, 0.0789, "... 192 floats ..."],
  "shape": [192]
}
```

---

## Audio Requirements

| Constraint | Value | What happens if violated |
|---|---|---|
| **Minimum duration** | 1.5 seconds | Returns `is_speaker: false, reason: "too_short"` |
| **Supported formats** | Any format ffmpeg can decode (.wav, .m4a, .mp3, .ogg, .flac, ...) | Returns HTTP 400 if undecoded |
| **Sample rate** | Any (auto-resampled to 16 kHz) | Handled by ffmpeg |
| **Channels** | Any (auto-converted to mono) | Handled by ffmpeg |

---

## Running Tests

```powershell
# Install test dependencies (first time only)
pip install httpx pytest

# Run all tests
python -m pytest tests/ -v

# Run with print output visible
python -m pytest tests/ -v -s
```

**Expected output:**
```
test_01_enroll_three_clips          PASSED     (enroll pipeline works)
test_02_verify_owner_voice          PASSED     (same-signal scores high)
test_03_verify_stranger_synthetic   XFAIL      (expected — synthetic audio can't simulate real speakers)
test_03b_verify_stranger_real_audio SKIPPED    (no stranger.wav provided)
test_04_verify_short_clip           PASSED     (short clips correctly rejected)
test_05_back_to_back_latency        PASSED     (both < 2s)

4 passed, 1 skipped, 1 xfailed
```

**To enable the real stranger test:** place a ≥2s audio recording from a different person at `tests/stranger.wav`.

---

## Project Structure

```
voice-service/
├── main.py                  # FastAPI app — /enroll, /verify, /health, /embed
├── speaker.py               # Model loading, embed(), compare()
├── store.py                 # Read/write owner embeddings (embeddings.json)
├── smoke_test.py            # Phase 0 verification script
├── requirements.txt         # Pinned Python dependencies
├── README.md                # This file
├── embeddings.json          # Owner voice data (created after first /enroll)
├── pretrained_models/
│   └── ecapa/               # Downloaded ECAPA-TDNN model (~200 MB)
├── tests/
│   ├── __init__.py
│   ├── test_speaker.py      # Phase 1 acceptance tests
│   └── stranger.wav         # (optional) real stranger audio for testing
└── venv/                    # Python virtual environment (not committed)
```

---

## Architecture Details

### Model: ECAPA-TDNN

- **Source:** [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Architecture:** ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)
- **Trained on:** VoxCeleb dataset (~7,000 speakers)
- **Output:** 192-dimensional float vector per utterance
- **Comparison:** Cosine similarity of L2-normalised embeddings

### Embedding Storage

Embeddings are stored in `embeddings.json`:

```json
{
  "shop_1_owner": {
    "mean": [0.012, -0.045, "... 192 floats ..."],
    "samples": [
      [0.011, -0.043, "..."],
      [0.013, -0.047, "..."],
      [0.012, -0.044, "..."]
    ]
  }
}
```

- **`mean`** — L2-normalised average of all enrolled samples. Used for verification.
- **`samples`** — Individual per-clip embeddings. Kept for threshold tuning in Phase 5.

The mean is recomputed and re-normalised every time a new sample is enrolled.

### Performance

| Operation | Target | Measured |
|---|---|---|
| Model load (warm) | < 60 s | ~3 s |
| Verify (warm model) | < 2 s | **0.6–0.9 s** |
| Back-to-back verifications | both < 2 s | **0.58 s, 0.69 s** |
| Embedding dimension | 192 | 192 |

---

## Re-enrollment

To re-enroll the owner (e.g., after a mic change or voice shift):

1. **Delete** `embeddings.json` in the `voice-service/` directory
2. **Restart** the service
3. **Enroll again** with 3–5 new clips via `POST /enroll`

```powershell
# Delete existing enrollment
Remove-Item embeddings.json

# Restart the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ffmpeg: command not found` | ffmpeg not on PATH | Install ffmpeg, add `bin/` to system PATH, restart terminal |
| `WinError 1314` (symlink privilege) | Windows blocks symlinks | Already handled — code uses `LocalStrategy.COPY_SKIP_CACHE` |
| `UnicodeEncodeError` on Windows | cp1252 terminal encoding | Already handled — `PYTHONUTF8=1` is set in code |
| First model load takes ~3 minutes | Downloading ~200 MB model | Expected on first run only. Subsequent loads are ~3s |
| `/verify` returns 404 | Speaker not enrolled | Call `/enroll` with 3+ clips first |
| Score is always high with synthetic audio | Model trained on speech, not sine waves | Use real voice recordings for meaningful results |
| Cold start `/verify` is slow (~4s) | Model JIT warmup | First call after boot is slower; subsequent calls are < 1s |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Suppress HuggingFace symlink warnings on Windows |
| `PYTHONUTF8` | `1` | Force UTF-8 encoding for Windows terminals |

---

## What's Next

| Phase | Description | Status |
|---|---|---|
| ~~Phase 0~~ | ~~Setup & environment~~ | ✅ Done |
| ~~Phase 1~~ | ~~Core service~~ | ✅ Done |
| **Phase 2** | Node backend integration | ⬜ Next |
| Phase 3 | Mobile app enrollment UI | ⬜ Pending |
| Phase 4 | Live gate (verification in production) | ⬜ Pending |
| Phase 5 | Threshold tuning with real shop data | ⬜ Pending |
| Phase 6 | On-device inference (optional) | ⬜ Later |

See [`docs/tasks.md`](../docs/tasks.md) for the full task checklist and [`docs/spec.md`](../docs/spec.md) for the technical specification.

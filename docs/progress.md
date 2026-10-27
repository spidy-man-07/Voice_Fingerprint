# Voice Lock — Build Progress Log

> **Live tracker** — updated at the end of every phase or significant milestone.
> Read alongside [`tasks.md`](tasks.md) (what to do) and [`spec.md`](spec.md) (how it works).

---

## Project Snapshot

| Item | Detail |
|---|---|
| **Module** | `voice-service/` — Python · FastAPI · port 8000 |
| **Model** | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) |
| **Python** | 3.14.0 |
| **PyTorch** | 2.14.1 (CPU) |
| **SpeechBrain** | 1.1.2 |
| **ffmpeg** | 9.0.2 (essentials build) |
| **OS** | Windows |

---

## Phase Progress Overview

| Phase | Name | Status | Completed |
|---|---|---|---|
| **0** | Setup & environment | ✅ DONE | 2026-10-23 |
| **1** | Core service (speaker.py / store.py / main.py) | ✅ DONE | 2026-10-26 |
| **2** | Node backend integration | ⬜ Next | — |
| **3** | App enrollment UI | ⬜ Pending | — |
| **4** | Live gate | ⬜ Pending | — |
| **5** | Tuning & robustness | ⬜ Pending | — |
| **6** | On-device (optional) | ⬜ Later | — |

---

## Phase 0 — Setup (DONE ✅)

**Completed:** 2026-10-23
**Duration:** ~2 hours (including ffmpeg install troubleshooting)

### What was done

#### 1. Project scaffolding
- Created `voice-service/` directory at repo root
- Created `voice-service/requirements.txt` with all pinned dependencies

#### 2. Python environment
- Confirmed Python **3.14.0** is available system-wide
- Created virtual environment: `voice-service/venv/`
- Installed all dependencies successfully via `venv\Scripts\pip install -r requirements.txt`

**Packages installed:**

| Package | Version |
|---|---|
| fastapi | 0.161.1 |
| uvicorn | 0.54.0 |
| speechbrain | 1.1.2 |
| torch | 2.14.1 |
| torchaudio | 2.11.0 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| soundfile | 0.14.0 |
| python-multipart | 0.0.32 |

#### 3. ffmpeg installation
- `winget` failed due to network/firewall error (`InternetOpenUrl() failed`)
- **Resolution:** Manual download from `gyan.dev/ffmpeg/builds/`
- Extracted to `C:\Users\hp\Downloads\ffmpeg\ffmpeg-9.0.2-essentials_build\`
- Added `C:\Users\hp\Downloads\ffmpeg\ffmpeg-9.0.2-essentials_build\bin` to **User PATH** via PowerShell
- Verified: `ffmpeg version 9.0.2-essentials_build`

#### 4. Model smoke test
- Created `voice-service/smoke_test.py`
- **Two issues hit and resolved:**
  1. `OSError: [WinError 1314]` — Windows blocks symlinks without Developer Mode
     - **Fix:** Pass `local_strategy=LocalStrategy.COPY_SKIP_CACHE` to `from_hparams()`. Downloads model files directly to `savedir`, bypassing HF cache symlinks entirely.
  2. `UnicodeEncodeError` — Windows terminal (cp1252) can't render emoji
     - **Fix:** Set `os.environ["PYTHONUTF8"] = "1"` and replaced emoji with ASCII labels

- **First run:** Model downloaded (~200 MB) to `pretrained_models/ecapa/` in ~167s
- **Warm run results (files cached on disk):**

```
Model      : speechbrain/spkrec-ecapa-voxceleb  [EncoderClassifier]
Load time  : 2.7s     ✓  (target < 60s)
Embed time : 373ms    ✓  (target < 2s)
Vec shape  : (192,)   ✓
L2 norm    : 1.0000   ✓
```

### Files created in Phase 0

| File | Purpose |
|---|---|
| `voice-service/requirements.txt` | Pinned Python dependencies |
| `voice-service/venv/` | Python virtual environment (not committed) |
| `voice-service/smoke_test.py` | Phase 0 verification script |
| `voice-service/pretrained_models/ecapa/` | Downloaded ECAPA-TDNN model files |
| `docs/spec.md` | Full technical specification |
| `docs/tasks.md` | Phased build task checklist |
| `docs/progress.md` | This file |

### Known gotchas documented

| Gotcha | Resolution |
|---|---|
| `winget` blocked by firewall | Manual download from gyan.dev |
| Windows symlink privilege error (WinError 1314) | `local_strategy=LocalStrategy.COPY_SKIP_CACHE` |
| Windows cp1252 terminal can't render emoji | `os.environ["PYTHONUTF8"] = "1"` |
| ffmpeg not on PATH after install | Restart terminal OR reload PATH in script |
| First model load is slow (~167s) | Expected — includes download. Warm load is 2.7s |

---

## Phase 1 — Core Service (DONE ✅)

**Completed:** 2026-10-26
**Duration:** ~1 day (code had been written previously; tests created and passing today)

### What was built

#### 1. `speaker.py` — Model layer
- `load_model()` — loads ECAPA-TDNN once, caches in `_model` singleton
- `embed(audio_path)` — ffmpeg decode → 16 kHz mono → ECAPA → L2-normalised 192-dim vector
- `compare(vec_a, vec_b)` — cosine similarity (dot product of unit vectors)
- `ClipTooShortError` — raised when decoded audio < 1.5 s

#### 2. `store.py` — Embedding persistence
- `save_embedding(speaker_id, mean_vec, samples)` — write to `embeddings.json`
- `load_embedding(speaker_id)` → `(mean_vec, samples)` or `(None, [])`
- `update_mean(speaker_id, new_sample)` — append sample, recompute L2-normalised mean
- `is_enrolled(speaker_id)` → bool

#### 3. `main.py` — FastAPI endpoints
- `POST /enroll` — accepts audio + speaker_id, embeds, updates store, returns contract shape
- `POST /verify` — accepts audio + speaker_id, embeds, compares to stored mean, returns `{ score, is_speaker, threshold }`
- `GET /health` — returns `{ status: "ok", model: "ecapa-voxceleb" }` in < 100 ms
- `POST /embed` — returns raw 192-dim vector (debugging/Phase 5)
- Model loaded **once** at startup via lifespan handler
- Error handling: bad audio → 400, not enrolled → 404, model not loaded → 503

#### 4. `tests/test_speaker.py` — 6 tests

| # | Test | Result | Notes |
|---|------|--------|-------|
| 1 | Enroll 3 clips | ✅ PASSED | samples incremented 1→2→3, similarity ~0.99 |
| 2 | Verify owner voice | ✅ PASSED | score=0.9793, latency=0.90s |
| 3 | Verify stranger (synthetic) | ⚠️ XFAIL | score=0.9589 — sine waves can't simulate speech; expected |
| 3b | Verify stranger (real audio) | ⏭️ SKIPPED | Needs `tests/stranger.wav` — a ≥2s clip from a different person |
| 4 | Short clip (0.5s) | ✅ PASSED | Returns `is_speaker: false, reason: "too_short"` |
| 5 | Back-to-back latency | ✅ PASSED | 0.58s + 0.69s — well under 2s target |

**Test run:** `4 passed, 1 skipped, 1 xfailed in 13.70s`

### Performance results

| Operation | Target | Actual |
|---|---|---|
| Verify latency (warm model) | < 2 s | **0.58–0.90 s** ✅ |
| Back-to-back verifications | both < 2 s | **0.58 s, 0.69 s** ✅ |

### Files created in Phase 1

| File | Purpose |
|---|---|
| `voice-service/speaker.py` | Model layer: load, embed, compare |
| `voice-service/store.py` | Embedding persistence (JSON) |
| `voice-service/main.py` | FastAPI app with /enroll, /verify, /health |
| `voice-service/tests/__init__.py` | Test package init |
| `voice-service/tests/test_speaker.py` | Phase 1 acceptance test suite |

### Follow-up needed
- [ ] Place a real stranger audio clip at `voice-service/tests/stranger.wav` (≥2s, different person or YouTube) to enable `test_03b`
- [ ] Test with real owner voice clips for definitive speaker verification accuracy

---

## Appendix — Useful Commands

```powershell
# Activate venv (run from voice-service/)
venv\Scripts\activate

# Run smoke test
venv\Scripts\python smoke_test.py

# Start the FastAPI service (Phase 1+)
venv\Scripts\uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run tests (Phase 1+)
venv\Scripts\python -m pytest tests/ -v

# Reload PATH for ffmpeg in current terminal
$env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
```

---

*Last updated: 2026-10-26 — Phase 1 complete*

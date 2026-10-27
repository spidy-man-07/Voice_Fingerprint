# Voice Lock — Build Tasks
**Assignee:** Voice-service teammate
**Service:** `voice-service/` · Python · FastAPI · port 8000
**Ask from project owner before starting:** laptop access · `.env` (Twilio/Sarvam, already configured) · one week of shop audio (for Phase 5)

---

## How We Proceed — The Big Picture

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  [Phase 6 optional]
 Setup      Core API    Node hook   App UI      Live gate   Tune & data   On-device
 0.5 day    1-2 days    1 day       1-2 days    1 day       2-3 days      later
```

Each phase has a concrete **"done when"** checkpoint. Do not start the next phase until the checkpoint passes. Phases 0–2 are purely yours. Phases 3–4 overlap with the mobile-app teammate.

---

## Phase 0 — Setup (Half a day)

**Goal:** Environment confirmed working, model loading verified.

### Tasks
- [ ] Create the `voice-service/` directory at the repo root (sibling of `backend/` and `mobile-app/`)
- [ ] Create `requirements.txt` with pinned versions (see `spec.md §10`)
- [ ] Create a Python virtual environment and install all dependencies
- [ ] Verify **ffmpeg** is installed: run `ffmpeg -version` in the service's terminal
  - If missing: install ffmpeg and add it to Windows PATH; confirm it works from the same terminal you'll run uvicorn from
- [ ] Write a one-liner smoke test: load the ECAPA model and print its class name
- [ ] Confirm the model downloads and loads in < 60 seconds

### Done When
- `ffmpeg -version` returns a version string in the service terminal
- The model loads once and stays in memory; a warm inference completes in < 2 s

---

## Phase 1 — Core Service (1–2 days)

**Goal:** All three API endpoints implemented and passing local tests.

### Tasks

#### `speaker.py` — Model layer
- [ ] Implement `load_model()` — load `speechbrain/spkrec-ecapa-voxceleb` once at module import, save to `pretrained_models/ecapa/`
- [ ] Implement `embed(wav_path: str) -> np.ndarray` — decode via ffmpeg subprocess, load 16 kHz mono PCM, run ECAPA, return L2-normalised 192-dim vector
- [ ] Implement `compare(vec_a, vec_b) -> float` — cosine similarity, return float in [−1, 1]
- [ ] Add minimum length check: if decoded audio < 1.5 s, raise a `ClipTooShortError`

#### `store.py` — Embedding persistence
- [ ] Implement `save_embedding(speaker_id, mean_vec, samples)` — write to `embeddings.json`
- [ ] Implement `load_embedding(speaker_id) -> (mean_vec, samples)` — read from `embeddings.json`
- [ ] Implement `update_mean(speaker_id, new_sample)` — append sample, recompute and re-normalise mean, overwrite file

#### `main.py` — FastAPI routes
- [ ] `POST /enroll` — accept `audio` + `speaker_id`, embed, update store, return contract shape (see `spec.md §5`)
- [ ] `POST /verify` — accept `audio` + `speaker_id`, embed, load stored mean, compare, return `{ score, is_speaker, threshold }`. Never 4xx/5xx on a low score.
- [ ] `GET /health` — return `{ "status": "ok", "model": "ecapa-voxceleb" }`. Respond in < 100 ms.
- [ ] Load model **once at startup** (`@app.on_event("startup")`), not per request
- [ ] Implement proper error handling: bad audio → 400, model not loaded → 503, all else → 500 with message

#### `tests/test_speaker.py`
- [ ] Test: enroll with 3 clips of your own voice
- [ ] Test: verify with a 4th clip of your own voice → expect `is_speaker: true`
- [ ] Test: verify with a YouTube clip or a different person → expect `is_speaker: false`
- [ ] Test: send a 0.5 s clip → expect `is_speaker: false, reason: "too_short"`

### Done When
- `python -m pytest tests/` passes all 4 tests from the command line
- Two back-to-back `/verify` calls each complete in < 2 s (warm model)

---

## Phase 2 — Node Backend Integration (1 day)

**Goal:** The existing Node/Express backend can call your service; it auto-starts with the project.

> **Note:** This phase touches `backend/src/` files — coordinate with the backend owner.

### Tasks (in `backend/src/`)

#### `services/voiceVerify.js` (NEW file)
- [ ] Create a small HTTP client that reads `VOICE_SERVICE_URL` from `.env` (default `http://127.0.0.1:8000`)
- [ ] Expose `verifyAudio(audioFilePath, speakerId)` → posts audio file, returns `{ is_speaker, score }`
- [ ] Expose `checkVoiceService()` → hits `/health`, logs result; called at Node startup

#### `routes/sale.js` — new route
- [ ] Add `POST /api/verify-and-transcribe` — same multipart body as `/api/transcribe`
  - Call `verifyAudio()` first
  - If `is_speaker: false` → return `{ text: "", speaker_verified: false }` with HTTP 200 (not an error)
  - If `is_speaker: true` → call existing `transcribeAudio()` → return its response with `speaker_verified: true` added
- [ ] Add `POST /api/enroll-voice` — forward to Python `/enroll`, return its JSON as-is
- [ ] Add `GET /api/voice-status` — call Python `/health` + check `embeddings.json` exists → return `{ enrolled: true/false }`

#### `server.js`
- [ ] Call `checkVoiceService()` at boot (log only, don't crash if down)

#### `.env` (document, do NOT commit)
- [ ] Add `VOICE_SERVICE_URL=http://127.0.0.1:8000`
- [ ] Add `VOICE_ENROLLED=true` (the app reads this to decide enrollment vs. normal mode)

#### `start-all.cmd` (repo root, NEW)
- [ ] Script that starts Node backend + Python voice-service + ngrok concurrently in one command

### Done When
- `curl -F audio=@me.wav http://127.0.0.1:3000/api/verify-and-transcribe` returns real Sarvam text for your voice
- The same curl with a stranger clip returns `{ "text": "" }`

---

## Phase 3 — App Enrollment UI (1–2 days)

**Goal:** A first-time owner can enroll from the phone without touching the laptop.

> **Note:** This phase is in `mobile-app/` — coordinate with the mobile-app teammate.

### Tasks (in `mobile-app/`)

#### `services/api.js`
- [ ] Add `verifyAndTranscribeAudio(fileUri)` — clone of `transcribeAudio` but hits `/api/verify-and-transcribe`
- [ ] Add `enrollVoice(fileUri)` — posts to `/api/enroll-voice`
- [ ] Add `getVoiceStatus()` — hits `/api/voice-status`, returns `{ enrolled: bool }`

#### `screens/HandsFreeScreen.js`
- [ ] On mount, call `getVoiceStatus()`
- [ ] If **not enrolled**: render enrollment panel above the mic button
  - Prompt: "Apni awaaz register karein — 3 baar 5-second bolna hoga"
  - Show a record button that calls `enrollVoice()` three times (show a different sentence each time)
  - After 3rd call, re-check status and hide the panel
- [ ] Replace `transcribeAudio(uri)` call with `verifyAndTranscribeAudio(uri)`
  - If `speaker_verified === false`: show brief toast "Awaaz pehchani nahi gayi" (~2 s), return — do NOT call `sendUtterance`, do NOT update `lastHeard`

### Done When
- A fresh install walks the owner through enrollment on the phone over Wi-Fi
- After enrollment, normal use works — owner's voice transcribed, stranger's voice silently ignored

---

## Phase 4 — Gate the Live Flow (1 day)

**Goal:** Verification failure silently gates the live `onSegment` path, with zero impact on a sale in progress.

### Tasks
- [ ] In `screens/HandsFreeScreen.js` → `onSegment`, ensure `verifyAndTranscribeAudio` is used (done in Phase 3)
- [ ] Confirm: rejected audio never reaches `sendUtterance`
- [ ] Confirm: rejected audio never shows in `lastHeard`
- [ ] Test in tap mode first (easier to control input)
- [ ] Test with music/TV playing in the background — transcript must stay empty for non-owner speech

### Done When
- Music/TV audio produces no items on the bill
- Owner's voice over the same music still adds items correctly

---

## Phase 5 — Tuning & Robustness (2–3 days, needs real data)

**Goal:** Threshold chosen with evidence; false-accept rate is as close to 0 as possible.

### Tasks
- [ ] Collect ~20 real audio clips at the shop:
  - 10 owner clips (varied: morning/evening, different distances, different sentences)
  - 10 non-owner clips (customers, family, radio/TV)
- [ ] Run a threshold sweep from 0.20 → 0.45 in steps of 0.05
- [ ] For each threshold, print a table: `threshold | false_accepts | false_rejects`
- [ ] Pick the threshold where `false_accept = 0` while `false_reject < 15%`
- [ ] Update the threshold in `main.py` and document the evidence table in `voice-service/README.md`
- [ ] Enable a JSONL verify log: every `/verify` call appends `{ timestamp, score, is_speaker }` (no audio stored)
- [ ] Run in production for one week; review the log before finalising the threshold

### Done When
- `voice-service/README.md` contains the evidence table with the chosen threshold and the sweep results

---

## Phase 6 — On-Device (Optional, Later)

**Goal:** Embed + compare runs on the phone; server becomes optional.

### Tasks
- [ ] Export the ECAPA model to TFLite (quantise to int8)
- [ ] Wrap in a small Expo module using `expo-dev-client`
- [ ] Run embed + compare on-device; keep server `/verify` as fallback

> **Decision gate:** Only pursue this if offline/no-server becomes a hard requirement for MVP. Skip for now.

---

## Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| ffmpeg not on PATH in Windows | Install, add to PATH, verify in the same terminal as uvicorn |
| Very short clips (< 1.5 s) | Enforce minimum in `/verify`; return `too_short` without scoring |
| Mic pipeline changed (Android `voice_recognition` source, noise-suppressed) | Enroll with the SAME pipeline the app uses — always enroll from the phone, never from a laptop mic file |
| Cold start (first `/verify` after boot takes 2–4 s) | Warm up at startup: POST 0.5 s silence to `/verify` |
| Two services on one laptop, phone on hotspot | Phone must reach laptop via Wi-Fi IP as `BASE_URL`. Windows Firewall: allow port 8000 on Private AND Public |
| False accept = stranger billed | Bias threshold upward; log every score for audit |

---

## Quick Reference

| What | Where |
|---|---|
| Full spec | `docs/spec.md` |
| Model API contracts | `docs/spec.md §5` |
| Architecture diagram | `docs/spec.md §2` |
| Done criteria per phase | This file, each phase section |
| Risk list (from roadmap) | This file, "Known Risks" table |
| Read order | spec §1 → §2 → §4 → §5 → Phase 0-2 here → spec §10 |

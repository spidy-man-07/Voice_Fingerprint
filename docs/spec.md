# Voice Lock — Technical Specification
**Module:** `voice-service/` (Python · FastAPI · port 8000)
**Role in the larger system:** Gatekeeper between the phone mic and Sarvam STT. Only the verified owner's audio ever reaches the billing pipeline.

---

## 1. The Core Idea in One Paragraph

The shop owner records ~15 seconds of their voice **once** (enrollment). The service converts that audio into a **192-dimensional voice fingerprint** (an embedding vector) using a pretrained ECAPA-TDNN model. From that point forward, every audio clip captured by the app is converted into its own embedding and compared against the stored owner embedding via **cosine similarity**. If the score clears the threshold → owner confirmed → audio forwarded to Sarvam STT. If not → audio silently dropped (and optionally a buzz notification sent).

> This is a **classification-threshold** problem, NOT a speech recognition problem. We do not train a model from scratch.

---

## 2. System Architecture

```
Phone (Expo app)
  │  audio file (.m4a)  ──── same upload path as today
  ▼
Node Backend (Express · port 3000)   [ALREADY EXISTS]
  POST /api/verify-and-transcribe  ──►  Python Voice Service (NEW · port 8000)
                                          ├── /embed    → audio → 192-dim vector
                                          ├── /compare  → two vectors → similarity score
                                          └── /enroll   → store owner's mean embedding
  │
  └── [if is_speaker: true] ──► Sarvam STT   [ALREADY EXISTS]
```

### Why Python microservice, not on-device?
| Concern | Reason |
|---|---|
| Model size | ECAPA-TDNN (~200 MB) would require TFLite conversion + RN integration — weeks of extra work |
| Network already required | Sarvam is a cloud API; the app is already network-first |
| Latency | embed + compare ≈ 150–300 ms CPU — acceptable beside the 1–2 s Sarvam call |
| Phase 6 (optional) | The same model can be exported to TFLite later if offline becomes a requirement |

---

## 3. Model Choice

**Model:** `speechbrain/spkrec-ecapa-voxceleb`
- Architecture: ECAPA-TDNN
- Pretrained on: VoxCeleb (~7,000 speakers)
- Output: 1 × 192 float vector per utterance
- Loaded via: `speechbrain.inference.speaker.EncoderClassifier`

**Comparison metric:** Cosine similarity (L2-normalised embeddings)
**Starting threshold:** 0.30 — tune upward in Phase 5 using real shop recordings

**Why not train our own model?**
We have exactly one positive speaker and zero negatives dataset. Pretrained embeddings + threshold is the industry-standard few-shot approach. Custom fine-tuning is a Phase 5+ experiment.

---

## 4. Audio Pipeline

```
Phone uploads .m4a (AAC)
        │
        ▼
ffmpeg subprocess (on the laptop running voice-service)
        │  ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav
        ▼
16 kHz mono PCM WAV  ──►  ECAPA-TDNN  ──►  192-dim embedding
```

**Critical constraint:** ffmpeg must be installed on the host machine AND available on `PATH` in the service's terminal environment. This is the **#1 thing that will break the first run** on Windows.

**Minimum clip length:** 1.5 seconds of actual audio post-decode. Clips shorter than this produce noisy embeddings → return `{ is_speaker: false, reason: "too_short" }` without scoring.

---

## 5. API Contracts (exact shapes — Node backend calls these)

### `POST /enroll`
- **Input:** `multipart/form-data` — field `audio` (m4a/wav), field `speaker_id` (use `"shop_1_owner"`)
- **Behaviour:** Call 3–5 times with different clips; stores the **mean embedding** (more stable than a single clip). Individual samples are also kept for later tuning.
- **Response:**
```json
{ "speaker_id": "shop_1_owner", "embedding_stored": true, "samples": 3, "similarity_mean": 0.82 }
```

### `POST /verify`
- **Input:** `multipart/form-data` — field `audio`, field `speaker_id`
- **Response:**
```json
{ "speaker_id": "shop_1_owner", "score": 0.41, "is_speaker": true, "threshold": 0.30 }
```
- `is_speaker = score >= threshold`. **Never throw** on "not the speaker" — that is a valid expected result. Only throw on real errors (bad audio, missing file, model not loaded).

### `GET /health`
- **Response:**
```json
{ "status": "ok", "model": "ecapa-voxceleb" }
```
- Used by the Node backend at boot (`checkVoiceService()`). Must respond quickly — model must be loaded **once at startup**, not per request.

---

## 6. Embedding Storage

- **Format:** JSON file (`store.py` manages this) or SQLite for Phase 2+
- **Location:** `voice-service/store.py`
- **Schema:** `{ speaker_id: string → mean_embedding: float[192], samples: float[][192] }`
- The mean embedding is re-normalised (L2) every time a new sample is added.

---

## 7. Performance Constraints

| Operation | Target |
|---|---|
| embed + compare (warm model) | < 2 seconds on laptop CPU |
| Two back-to-back verifications | both < 2 s |
| Service startup (model load) | < 60 seconds |
| Cold start warm-up | Call `/verify` with 0.5 s silence at startup to force lazy init |

---

## 8. Failure & UX Rules

| Failure mode | Classification | Handling |
|---|---|---|
| Stranger billed (false accept) | **Catastrophic** | Threshold must bias toward rejection here |
| Owner ignored (false reject) | Annoying but safe | App shows "not recognised" hint; owner taps again |
| Voice service down | Serious | Fail open WITH a warning (log loudly); never silently stop billing |
| Clip too short | Expected | Return `is_speaker: false, reason: "too_short"` |
| Re-enrollment needed | Operational | Delete stored embedding file; re-run enrollment flow |

> **UX Rule (non-negotiable):** Verification failure must **never block** the countdown→send flow mid-sale. If items are on the bill and the next utterance fails verification, just don't add anything. Don't reset, don't error-out.

---

## 9. Files This Module Owns

```
voice-service/
├── requirements.txt       # pinned versions
├── main.py                # FastAPI app — /embed, /verify, /enroll, /health
├── speaker.py             # model loading, embed(), compare() logic
├── store.py               # read/write owner embeddings (JSON or SQLite)
└── tests/
    └── test_speaker.py    # enroll 3 clips → verify 4th (own voice) → verify stranger
```

---

## 10. Dependencies (`requirements.txt`)

```
fastapi
uvicorn[standard]
speechbrain
torch              # CPU wheel is fine; ~200 MB
torchaudio
python-multipart   # file uploads
numpy
scipy              # cosine similarity
soundfile          # read wav after ffmpeg decode
```

---

## 11. What "Done" Looks Like (End-to-End Demo)

1. Fresh app → enrollment panel → owner records 3 sentences → embeddings stored.
2. Music playing in background → customer speaks "5 kilo cheeni teen sau rupaye" → nothing appears on the bill.
3. Owner speaks the same sentence → item appears on the bill.
4. Owner dictates the customer's number → countdown starts → bill generated.
5. Voice-service log shows verify scores for every clip with timestamps.

---

*Read order for this module's builder:* §1 → §2 → §4 → §5 → Phase 0-2 in tasks.md → §10 risks.

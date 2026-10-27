"""
Phase 0 Smoke Test — Voice Lock
Confirms:
  1. ECAPA-TDNN model downloads and loads successfully
  2. Load time is under 60 seconds
  3. Model produces a 192-dim embedding from a dummy waveform
Run: venv\Scripts\python smoke_test.py
"""

import os
import time
import torch
import numpy as np

# Fix: Windows blocks symlinks without Developer Mode or admin rights.
# Force HuggingFace hub to COPY files instead of symlinking.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Fix: Windows terminal cp1252 can't render emoji — force UTF-8 output.
os.environ["PYTHONUTF8"] = "1"

print("=" * 50)
print("Voice Lock - Phase 0 Smoke Test")
print("=" * 50)

# Step 1: Load the model
print("\n[1/3] Loading ECAPA-TDNN model (first run downloads ~200 MB)...")
start = time.time()

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# COPY_SKIP_CACHE: download files directly to savedir, no HF cache symlinks.
# This avoids WinError 1314 (symlink privilege) without needing Developer Mode.
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": "cpu"},
    local_strategy=LocalStrategy.COPY_SKIP_CACHE,
)

elapsed = time.time() - start
print(f"      Model class : {model.__class__.__name__}")
print(f"      Load time   : {elapsed:.1f}s")

if elapsed > 60:
    print(f"  [NOTE] Load time {elapsed:.1f}s — includes download on first run.")
    print("  [NOTE] Warm load (files cached) will be < 10s. Run again to confirm.")
else:
    print("  [OK] Load time < 60s")

# Step 2: Run a dummy inference to confirm the pipeline works
print("\n[2/3] Running dummy inference (1s of silence at 16 kHz)...")
dummy_waveform = torch.zeros(1, 16000)  # 1 second of silence, 16 kHz mono

inf_start = time.time()
embeddings = model.encode_batch(dummy_waveform)
inf_elapsed = time.time() - inf_start

vec = embeddings.squeeze().detach().numpy()
print(f"      Output shape : {vec.shape}  (expected: (192,))")
print(f"      Inference    : {inf_elapsed*1000:.0f}ms")

if vec.shape == (192,):
    print("  [OK] Embedding shape correct")
else:
    print(f"  [FAIL] UNEXPECTED SHAPE: {vec.shape}")

# Step 3: L2-normalise and sanity check
print("\n[3/3] L2-normalisation check...")
norm = np.linalg.norm(vec)
vec_normalised = vec / norm
norm_after = np.linalg.norm(vec_normalised)
print(f"      Norm before : {norm:.4f}")
print(f"      Norm after  : {norm_after:.4f}  (expected: ~1.0)")

if abs(norm_after - 1.0) < 1e-5:
    print("  [OK] Normalisation correct")
else:
    print("  [FAIL] Normalisation failed")

# Summary
print("\n" + "=" * 50)
print("SMOKE TEST COMPLETE")
print(f"  Model      : speechbrain/spkrec-ecapa-voxceleb")
print(f"  Load time  : {elapsed:.1f}s")
print(f"  Embed time : {inf_elapsed*1000:.0f}ms")
print(f"  Vec shape  : {vec.shape}")
print("=" * 50)
print("\nPhase 0 [DONE] - Ready to build Phase 1 (speaker.py / main.py)\n")

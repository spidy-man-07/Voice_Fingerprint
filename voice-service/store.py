"""
store.py — Voice Lock
Embedding persistence layer: read/write owner voice embeddings to embeddings.json.

Schema (embeddings.json)
------------------------
{
  "<speaker_id>": {
    "mean":    [float, ...],        # 192-dim L2-normalised mean embedding
    "samples": [[float, ...], ...]  # individual per-clip embeddings (kept for tuning)
  }
}

Public API
----------
save_embedding(speaker_id, mean_vec, samples)   -> None
load_embedding(speaker_id)                       -> (mean_vec, samples) or (None, [])
update_mean(speaker_id, new_sample)              -> (mean_vec, samples)
is_enrolled(speaker_id)                          -> bool
"""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store location — same directory as this file
# ---------------------------------------------------------------------------

STORE_PATH = Path(__file__).parent / "embeddings.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_store() -> dict:
    """Read the full JSON store from disk. Returns empty dict if file absent."""
    if STORE_PATH.exists():
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_store(data: dict) -> None:
    """Atomically overwrite the JSON store."""
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.debug("Store written to %s", STORE_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_embedding(
    speaker_id: str,
    mean_vec: np.ndarray,
    samples: list,
) -> None:
    """
    Persist a speaker's mean embedding and individual samples to disk.

    Parameters
    ----------
    speaker_id : str
        Unique identifier, e.g. "shop_1_owner".
    mean_vec : np.ndarray  shape (192,)
        L2-normalised mean of all samples.
    samples : list of np.ndarray shape (192,)
        Individual per-clip embeddings. Kept for threshold tuning in Phase 5.
    """
    data = _load_store()
    data[speaker_id] = {
        "mean": mean_vec.tolist(),
        "samples": [s.tolist() for s in samples],
    }
    _write_store(data)
    logger.info(
        "Saved embedding for '%s' — %d sample(s) stored.", speaker_id, len(samples)
    )


def load_embedding(speaker_id: str):
    """
    Load a speaker's mean embedding and individual samples from disk.

    Returns
    -------
    (mean_vec, samples)
        mean_vec : np.ndarray shape (192,), or None if speaker not enrolled
        samples  : list of np.ndarray shape (192,), empty list if not enrolled
    """
    data = _load_store()
    if speaker_id not in data:
        logger.warning("Speaker '%s' not found in store.", speaker_id)
        return None, []

    entry = data[speaker_id]
    mean_vec = np.array(entry["mean"], dtype=np.float32)
    samples = [np.array(s, dtype=np.float32) for s in entry["samples"]]
    logger.debug(
        "Loaded embedding for '%s' — %d sample(s).", speaker_id, len(samples)
    )
    return mean_vec, samples


def update_mean(speaker_id: str, new_sample: np.ndarray):
    """
    Append a new clip embedding to the speaker's sample list, recompute the
    mean, re-normalise it (L2), and persist everything to disk.

    This is called by /enroll for each new audio clip. Using the mean of
    multiple clips is more stable than any single recording.

    Parameters
    ----------
    speaker_id  : str
    new_sample  : np.ndarray shape (192,), already L2-normalised

    Returns
    -------
    (mean_vec, samples)
        mean_vec : np.ndarray shape (192,) — updated L2-normalised mean
        samples  : list of np.ndarray — all samples including the new one
    """
    _, existing_samples = load_embedding(speaker_id)
    samples = existing_samples + [new_sample]

    # Stack all samples and compute mean
    stacked = np.stack(samples, axis=0)        # (N, 192)
    raw_mean = stacked.mean(axis=0)            # (192,)

    # Re-normalise — mean of unit vectors is NOT necessarily unit-norm
    norm = np.linalg.norm(raw_mean)
    mean_vec = raw_mean / norm if norm > 1e-8 else raw_mean

    save_embedding(speaker_id, mean_vec, samples)
    logger.info(
        "Updated mean for '%s' with %d total sample(s). "
        "New mean norm: %.4f",
        speaker_id, len(samples), float(np.linalg.norm(mean_vec)),
    )
    return mean_vec, samples


def is_enrolled(speaker_id: str) -> bool:
    """Return True if the speaker has at least one stored embedding."""
    data = _load_store()
    return speaker_id in data

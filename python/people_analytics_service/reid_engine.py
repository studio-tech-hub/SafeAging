"""S P2.1 — Person Re-Identification feature extractor.

Uses MobileNetV3-Small (torchvision, CPU-friendly) as a body-appearance backbone.
Produces a 576-dim L2-normalised float32 embedding that can be stored compactly
(576 × 4 = 2 304 bytes) and compared via cosine similarity.

Design notes
------------
- Lazy model load on first call; cached in module-level variable (thread-safe).
- All public functions are synchronous and CPU-only so they can be called via
  `asyncio.to_thread()` from async contexts without blocking the event loop.
- Embeddings are stored as raw float32 bytes in the `person_embeddings` table.
"""

import logging
import threading
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvmodels
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 576
DEFAULT_MATCH_THRESHOLD: float = 0.65  # cosine similarity; tune to reduce false positives

_lock = threading.Lock()
_extractor: Optional[nn.Module] = None


def _build_extractor() -> nn.Module:
    weights = tvmodels.MobileNet_V3_Small_Weights.DEFAULT
    base = tvmodels.mobilenet_v3_small(weights=weights)
    # features → avgpool → flatten  =  576-dim body embedding
    extractor = nn.Sequential(base.features, base.avgpool, nn.Flatten(1))
    extractor.eval()
    return extractor


def _get_extractor() -> nn.Module:
    global _extractor
    if _extractor is None:
        with _lock:
            if _extractor is None:
                logger.info("[reid] Loading MobileNetV3-Small (dim=%d)…", EMBEDDING_DIM)
                _extractor = _build_extractor()
                logger.info("[reid] Feature extractor ready")
    return _extractor


def _preprocess(bgr: np.ndarray) -> torch.Tensor:
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty crop passed to reid_engine")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    tensor = TF.to_tensor(pil)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor.unsqueeze(0)  # (1, 3, 224, 224)


def compute_embedding(bgr_crop: np.ndarray) -> np.ndarray:
    """Extract a 576-dim L2-normalised body embedding from a BGR numpy crop.

    Thread-safe, CPU-only, deterministic.  Expects the crop to be a single
    person's bounding-box region (not the full frame).

    Parameters
    ----------
    bgr_crop : np.ndarray
        BGR image array of any size (will be resized to 224×224 internally).

    Returns
    -------
    np.ndarray
        float32 vector of shape (576,), L2-normalised.
    """
    tensor = _preprocess(bgr_crop)
    extractor = _get_extractor()
    with torch.no_grad():
        feat = extractor(tensor)
    vec = feat.squeeze().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Serialise a float32 embedding to raw bytes for DB storage."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialise raw bytes back to a float32 numpy array."""
    return np.frombuffer(data, dtype=np.float32).copy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Assumes L2-normalised inputs (dot product only)."""
    return float(np.dot(a, b))


def find_best_match(
    query: np.ndarray,
    candidates: list[tuple],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> tuple | None:
    """Return (id, score) of the best matching candidate above threshold, else None.

    Parameters
    ----------
    query : np.ndarray
        Query embedding (L2-normalised, float32).
    candidates : list of (id, np.ndarray)
        Reference embeddings to compare against.
    threshold : float
        Minimum cosine similarity for a positive match.
    """
    if not candidates:
        return None
    best_id, best_score = None, -1.0
    for cid, ref in candidates:
        s = cosine_sim(query, ref)
        if s > best_score:
            best_score, best_id = s, cid
    if best_id is not None and best_score >= threshold:
        return best_id, best_score
    return None

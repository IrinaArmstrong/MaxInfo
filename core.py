"""Core MaxInfo algorithm operating on pre-computed frame features.

This module depends on NumPy only (no torch / transformers), so it can be used
with any frame encoder.

Pipeline (see :func:`select_key_frame_indices`)::

    features [T, D]
      -> SVD, keep first ``r`` left singular vectors      [T, r]
      -> rect_maxvol (max-volume rows)                     indices
      -> sort
      -> (optional) drop frames too similar to the previous kept one
      -> (optional) drop the last frame if the count is odd
      -> (optional) uniformly sub-sample down to ``max_frames``
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ._maxvol_backend import rect_maxvol

__all__ = [
    "svd_reduce_dim",
    "max_vol_extract_index",
    "filter_similar_frames",
    "select_key_frame_indices",
]


def svd_reduce_dim(features: np.ndarray, r: int = 8) -> np.ndarray:
    """Project frame features onto their first ``r`` left singular vectors.

    Args:
        features: Array of shape ``[T, D]``; one row per frame.
        r: Number of singular vectors to keep. Clipped to ``min(T, D)``.

    Returns:
        Array of shape ``[T, min(r, T, D)]`` that can be fed directly to maxvol.
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
    u, _, _ = np.linalg.svd(features, full_matrices=False)
    return u[:, : min(int(r), u.shape[1])]


def max_vol_extract_index(low_dim: np.ndarray, tol: float = 0.3) -> np.ndarray:
    """Pick the rows of ``low_dim`` that (approximately) maximise the volume.

    Args:
        low_dim: Output of :func:`svd_reduce_dim`, shape ``[T, r]``.
        tol: ``rect_maxvol`` tolerance. Smaller values -> more rows selected.

    Returns:
        Array of selected row (frame) indices, in pivot order (not sorted).
    """
    piv, _ = rect_maxvol(low_dim, tol)
    return np.asarray(piv)


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    denom = max(np.linalg.norm(a) * np.linalg.norm(b), eps)
    return float(np.dot(a, b) / denom)


def filter_similar_frames(
    features: np.ndarray,
    selected_indices: Sequence[int],
    similarity_threshold: float = 0.99,
) -> List[int]:
    """Drop selected frames that are near-duplicates of the previously kept one.

    Frames are visited in the given order (pass *sorted* indices); a frame is
    kept iff its cosine similarity to the last *kept* frame is strictly below
    ``similarity_threshold``. The first frame is always kept.

    Args:
        features: Original (not SVD-reduced) features, shape ``[T, D]``.
        selected_indices: Indices into ``features``.
        similarity_threshold: Cosine-similarity cut-off in ``[-1, 1]``. A value
            ``> 1`` keeps everything.

    Returns:
        The kept indices (as a list of ints).
    """
    selected_indices = [int(i) for i in selected_indices]
    if not selected_indices:
        return []
    features = np.asarray(features)
    kept = [selected_indices[0]]
    for idx in selected_indices[1:]:
        if _cosine_similarity(features[kept[-1]], features[idx]) < similarity_threshold:
            kept.append(idx)
    return kept


def select_key_frame_indices(
    features: np.ndarray,
    *,
    r: int = 8,
    tol: float = 0.3,
    filter_similar: bool = True,
    similarity_threshold: float = 0.99,
    even_frames: bool = True,
    max_frames: Optional[int] = 50,
) -> np.ndarray:
    """Run the full MaxInfo selection on frame features.

    Args:
        features: ``[T, D]`` per-frame embeddings (e.g. CLIP image features).
        r: SVD rank used before maxvol.
        tol: ``rect_maxvol`` tolerance.
        filter_similar: Enable the cosine-similarity de-duplication step.
        similarity_threshold: Threshold for that step (ignored if disabled).
        even_frames: If more than one frame is selected and the count is odd,
            drop the last one (some video models need an even frame count).
        max_frames: If set and more frames are selected, uniformly sub-sample
            the selection down to this many frames. ``None`` disables the cap.

    Returns:
        Sorted ``np.ndarray`` of selected frame indices (dtype ``int64``).
    """
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if features.dtype != np.float32 and features.dtype != np.float64:
        features = features.astype(np.float32)

    low_dim = svd_reduce_dim(features, r)
    selections = np.sort(max_vol_extract_index(low_dim, tol)).astype(np.int64)

    if filter_similar:
        selections = np.asarray(
            filter_similar_frames(features, selections, similarity_threshold),
            dtype=np.int64,
        )

    if even_frames and len(selections) > 1 and len(selections) % 2 != 0:
        selections = selections[:-1]

    if max_frames is not None and len(selections) > max_frames:
        keep = np.linspace(0, len(selections) - 1, max_frames).astype(np.int64)
        selections = selections[keep]

    return selections

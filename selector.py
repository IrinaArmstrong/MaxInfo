"""High-level MaxInfo key-frame selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .core import select_key_frame_indices
from .encoder import DEFAULT_VISION_MODEL, ClipFrameEncoder
from .frames import take_frames

__all__ = ["MaxInfo", "MaxInfoResult"]


@dataclass
class MaxInfoResult:
    """Output of :meth:`MaxInfo.select`."""

    indices: np.ndarray       # sorted indices of the selected frames
    num_candidates: int       # number of input frames
    features: np.ndarray      # [T, D] features of *all* input frames

    def __len__(self) -> int:
        return len(self.indices)


class MaxInfo:
    """Training-free key-frame selection with the maximum-volume principle.

    Args:
        vision_model_path: HF id / local path of the CLIP-like encoder.
        encoder: A custom encoder object with ``encode(frames) -> np.ndarray[T, D]``.
            If given, ``vision_model_path`` / ``batch_size`` / ``device`` are
            ignored. Handy to share one CLIP across experiments.
        max_frames: Upper bound on returned frames (uniform sub-sampling of the
            selection). ``None`` -> no cap.
        batch_size: Frames per encoder forward pass.
        device: Device for the built-in encoder (default: cuda if available).
        r: SVD rank used before maxvol.
        tol: ``rect_maxvol`` tolerance (smaller -> more frames selected).
        filter_similar: Turn the cosine-similarity de-duplication on/off.
        similarity_threshold: Cosine-similarity threshold of that filter: a
            selected frame is dropped if its similarity to the previously kept
            frame is >= threshold. Ignored when ``filter_similar`` is False.
        even_frames: Drop the last selected frame if the count is odd.

    Example::

        selector = MaxInfo(max_frames=64, filter_similar=False)
        idx = selector.select_indices(frames)          # np.ndarray of ints
        key_frames = selector.select_frames(frames)    # same container type
    """

    def __init__(
        self,
        vision_model_path: str = DEFAULT_VISION_MODEL,
        encoder: Optional[Any] = None,
        max_frames: Optional[int] = 50,
        batch_size: int = 32,
        device: Optional[str] = None,
        r: int = 8,
        tol: float = 0.3,
        filter_similar: bool = True,
        similarity_threshold: float = 0.99,
        even_frames: bool = True,
    ):
        self.encoder = encoder if encoder is not None else ClipFrameEncoder(
            model_path=vision_model_path, batch_size=batch_size, device=device
        )
        self.max_frames = max_frames
        self.r = r
        self.tol = tol
        self.filter_similar = filter_similar
        self.similarity_threshold = similarity_threshold
        self.even_frames = even_frames

    # ------------------------------------------------------------------
    def select_from_features(self, features: np.ndarray) -> np.ndarray:
        """Select frame indices from precomputed ``[T, D]`` features."""
        return select_key_frame_indices(
            features,
            r=self.r,
            tol=self.tol,
            filter_similar=self.filter_similar,
            similarity_threshold=self.similarity_threshold,
            even_frames=self.even_frames,
            max_frames=self.max_frames,
        )

    def select(self, frames: Any, features: Optional[np.ndarray] = None) -> MaxInfoResult:
        """Select key frames and return indices together with the features.

        Args:
            frames: List of frames (PIL / HWC or CHW arrays / tensors) or a 4-D
                array/tensor ``[T,H,W,C]`` / ``[T,C,H,W]``.
            features: Optionally pass precomputed features to skip encoding.
        """
        if features is None:
            features = self.encoder.encode(frames)
        features = np.asarray(features)
        indices = self.select_from_features(features)
        return MaxInfoResult(indices=indices, num_candidates=len(features), features=features)

    def select_indices(self, frames: Any, features: Optional[np.ndarray] = None) -> np.ndarray:
        """Return only the sorted indices of the selected frames."""
        return self.select(frames, features).indices

    def select_frames(self, frames: Any, features: Optional[np.ndarray] = None) -> Any:
        """Return the selected frames, in the same container type as the input."""
        return take_frames(frames, self.select_indices(frames, features))

    __call__ = select_indices

    def __repr__(self) -> str:
        return (
            f"MaxInfo(max_frames={self.max_frames}, r={self.r}, tol={self.tol}, "
            f"filter_similar={self.filter_similar}, "
            f"similarity_threshold={self.similarity_threshold}, even_frames={self.even_frames})"
        )

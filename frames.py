"""Helpers to normalise the many ways a "video" can be represented."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

__all__ = ["to_pil_frames", "take_frames"]


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor without importing torch
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_to_pil(frame: Any, channels_last: Optional[bool]) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    arr = _to_numpy(frame)
    if arr.ndim == 2:  # grayscale H x W
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Cannot interpret frame with shape {arr.shape}")

    if channels_last is None:
        # Heuristic: channel axis is the one of size 1/3/4 (prefer channels-last).
        if arr.shape[-1] in (1, 3, 4):
            channels_last = True
        elif arr.shape[0] in (1, 3, 4):
            channels_last = False
        else:
            raise ValueError(
                f"Cannot infer channel axis for frame shape {arr.shape}; "
                "pass channels_last explicitly."
            )
    if not channels_last:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() <= 1.0:  # assume [0, 1] floats
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


def to_pil_frames(frames: Any, channels_last: Optional[bool] = None) -> List[Image.Image]:
    """Convert a video into a list of RGB PIL images.

    Accepted inputs:
      * a list/tuple of ``PIL.Image`` / ``np.ndarray`` / ``torch.Tensor`` frames;
      * a 4-D ``np.ndarray`` / ``torch.Tensor``: ``[T, H, W, C]`` (e.g. decord)
        or ``[T, C, H, W]``.

    Args:
        frames: The video.
        channels_last: ``True`` for HWC, ``False`` for CHW, ``None`` to infer.
            Float arrays are assumed to be in [0, 1] if their max is <= 1,
            otherwise in [0, 255].
    """
    if isinstance(frames, (list, tuple)):
        return [_frame_to_pil(f, channels_last) for f in frames]
    arr = _to_numpy(frames)
    if arr.ndim != 4:
        raise ValueError(
            f"Expected a list of frames or a 4-D array [T,H,W,C]/[T,C,H,W], got shape {arr.shape}"
        )
    return [_frame_to_pil(f, channels_last) for f in arr]


def take_frames(frames: Any, indices: Sequence[int]) -> Any:
    """Index a video container with ``indices``, preserving its type.

    Lists/tuples return a list; arrays/tensors are indexed along dim 0.
    """
    indices = [int(i) for i in indices]
    if isinstance(frames, (list, tuple)):
        return [frames[i] for i in indices]
    return frames[indices]

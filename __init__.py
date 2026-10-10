"""MaxInfo: training-free key-frame selection via the maximum-volume principle.

Quick start::

    from maxinfo import MaxInfo

    selector = MaxInfo(max_frames=64, filter_similar=False)
    key_idx = selector.select_indices(frames)
"""

from .core import (
    filter_similar_frames,
    max_vol_extract_index,
    select_key_frame_indices,
    svd_reduce_dim,
)
from .encoder import DEFAULT_VISION_MODEL, ClipFrameEncoder
from .selector import MaxInfo, MaxInfoResult

__version__ = "0.1.0"

__all__ = [
    "MaxInfo",
    "MaxInfoResult",
    "ClipFrameEncoder",
    "DEFAULT_VISION_MODEL",
    "select_key_frame_indices",
    "svd_reduce_dim",
    "max_vol_extract_index",
    "filter_similar_frames",
]

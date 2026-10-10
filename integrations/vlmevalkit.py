"""Adapter for VLMEvalKit-style messages.

A message is a list of dicts like ``{"type": "image" | "text", "value": ...}``
where image values are frame file paths.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from PIL import Image

from ..selector import MaxInfo

__all__ = ["filter_message_images", "apply_maxinfo_to_message"]


def filter_message_images(message: List[Dict[str, str]], keep_ids: Sequence[int]) -> List[Dict[str, str]]:
    """Keep all text items and only the images whose position is in ``keep_ids``."""
    keep = set(int(i) for i in keep_ids)
    out, image_cnt = [], 0
    for item in message:
        if item["type"] == "image":
            if image_cnt in keep:
                out.append(item)
            image_cnt += 1
        else:
            out.append(item)
    return out


def apply_maxinfo_to_message(selector: MaxInfo, message: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Run ``selector`` on the image items of ``message`` and drop the rest."""
    paths = [x["value"] for x in message if x["type"] == "image"]
    frames = [Image.open(p).convert("RGB") for p in paths]
    ids = selector.select_indices(frames)
    print(f"[maxinfo] selected {len(ids)} frames from {len(frames)}.")
    return filter_message_images(message, ids)

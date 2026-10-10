"""CLIP-style frame encoder (lazy: torch / transformers are imported on use)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .frames import to_pil_frames

__all__ = ["ClipFrameEncoder", "DEFAULT_VISION_MODEL"]

DEFAULT_VISION_MODEL = "openai/clip-vit-large-patch14-336"


class ClipFrameEncoder:
    """Turns frames into feature vectors with a CLIP-like ``get_image_features``.

    The model is loaded lazily on first use, so constructing the encoder is
    cheap. You may also inject an already-loaded ``model`` / ``processor`` to
    avoid loading CLIP a second time.

    Args:
        model_path: HF id or local path (used only if ``model`` is None).
        model: Optional preloaded model exposing ``get_image_features``.
        processor: Optional preloaded image processor (required with ``model``
            unless it can be loaded from ``model_path``).
        device: ``"cuda"``, ``"cpu"``, ``"cuda:1"``, ... Defaults to cuda if
            available. Ignored for an injected ``model`` (its own device is used).
        batch_size: Number of frames per forward pass.
        dtype: Optional torch dtype (e.g. ``torch.float16``) for a model loaded
            from ``model_path``.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_VISION_MODEL,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        dtype: Optional[Any] = None,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype

    # -- lazy loading ----------------------------------------------------
    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if self._model is None:
            kwargs = {"torch_dtype": self._dtype} if self._dtype is not None else {}
            self._model = AutoModel.from_pretrained(self.model_path, **kwargs)
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device).eval()
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path)

    @property
    def model(self):
        if self._model is None or self._processor is None:
            self._load()
        return self._model

    @property
    def processor(self):
        if self._model is None or self._processor is None:
            self._load()
        return self._processor

    @property
    def device(self):
        return next(self.model.parameters()).device

    # -- encoding --------------------------------------------------------
    def encode(self, frames: Any, channels_last: Optional[bool] = None) -> np.ndarray:
        """Encode frames to a float32 array of shape ``[T, D]``."""
        import torch

        pil_frames = to_pil_frames(frames, channels_last)
        model, processor = self.model, self.processor
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        feats = []
        with torch.no_grad():
            for i in range(0, len(pil_frames), self.batch_size):
                batch = pil_frames[i : i + self.batch_size]
                inputs = processor(images=batch, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(device=device, dtype=model_dtype)
                out = model.get_image_features(pixel_values=pixel_values)
                if not isinstance(out, torch.Tensor):  # newer transformers may wrap the output
                    out = getattr(out, "pooler_output", out[0])
                feats.append(out.float().cpu())
        if not feats:
            return np.zeros((0, 0), dtype=np.float32)
        return torch.cat(feats, dim=0).numpy()

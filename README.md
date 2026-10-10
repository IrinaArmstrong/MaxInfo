# maxinfo – MaxInfo as a reusable module

Training-free key-frame selection with the maximum-volume principle
(CLIP features → SVD → `rect_maxvol` → optional cosine de-duplication → even trim → cap).
Based on `VLMEvalKit/vlmeval/utils/maxinfo_reduction.py`, with the cosine filter made optional.

## Install / import

```python
# Option A: no install – put the parent folder on sys.path (or PYTHONPATH)
import sys; sys.path.insert(0, "/home/jovyan/abdullaeva-nfs2/video-llm/MaxInfo")
from maxinfo import MaxInfo

# Option B: pip install /home/jovyan/abdullaeva-nfs2/video-llm/MaxInfo/maxinfo[clip]
```

Requirements: `numpy`, `scipy`, `pillow`; `torch` + `transformers` only for the built-in CLIP encoder.
`rect_maxvol` comes from an installed `maxvolpy` if available (compiled, faster),
otherwise from the pure-Python copy in `_vendor/maxvolpy`.

## Usage

```python
from maxinfo import MaxInfo

selector = MaxInfo(
    max_frames=64,               # cap (None = no cap)
    r=8, tol=0.3,                # SVD rank / maxvol tolerance
    filter_similar=True,         # cosine de-duplication on/off
    similarity_threshold=0.99,   # drop a frame if cos-sim to previous kept frame >= this
    even_frames=True,            # drop last frame if count is odd
)

idx   = selector.select_indices(frames)   # sorted np.ndarray of ints
kept  = selector.select_frames(frames)    # same container type as `frames`
res   = selector.select(frames)           # .indices, .features, .num_candidates
```

`frames` can be a list of PIL images / HWC or CHW arrays / tensors, or a 4-D array/tensor
`[T,H,W,C]` / `[T,C,H,W]`.

Reproducing the official notebooks: `filter_similar=False` (they never use the filter);
`tol=0.1` for the LLaVA/Qwen notebooks (they hardcode it), `tol=0.3` for MiniCPM;
`max_frames=64` (Slow version) or `None` (Fast version).

### Reuse an already-loaded CLIP, or any other encoder

```python
from maxinfo import MaxInfo, ClipFrameEncoder
enc = ClipFrameEncoder(model=clip_model, processor=clip_processor, batch_size=16)
selector = MaxInfo(encoder=enc)        # or any object with .encode(frames) -> np.ndarray [T, D]
```

### Features only (no model, NumPy only)

```python
from maxinfo import select_key_frame_indices
idx = select_key_frame_indices(features, filter_similar=False, max_frames=64)
```

### VLMEvalKit messages

```python
from maxinfo.integrations import apply_maxinfo_to_message
message = apply_maxinfo_to_message(selector, message)
```

## Layout

```
maxinfo/
  core.py          # numpy-only algorithm (svd, maxvol, filter, full pipeline)
  encoder.py       # lazy CLIP encoder
  frames.py        # frame-format helpers
  selector.py      # MaxInfo class
  integrations/    # VLMEvalKit message adapter
  _maxvol_backend.py, _vendor/maxvolpy/   # rect_maxvol (MIT, A. Mikhalev)
```

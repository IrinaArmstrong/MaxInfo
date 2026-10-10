"""Resolve ``rect_maxvol``.

Prefers an installed ``maxvolpy`` (which ships a compiled, fast implementation)
and falls back to the pure-Python copy vendored in ``maxinfo/_vendor``.
"""

try:
    from maxvolpy.maxvol import rect_maxvol  # type: ignore
    BACKEND = "maxvolpy (installed)"
except ImportError:
    from ._vendor.maxvolpy.maxvol import rect_maxvol
    BACKEND = "maxinfo._vendor.maxvolpy"

__all__ = ["rect_maxvol", "BACKEND"]

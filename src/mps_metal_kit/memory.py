"""Unified Memory watermark configuration and zero-swap cache management for Apple Silicon MPS."""
from __future__ import annotations

import os
import torch


def configure_mps_memory(
    low_watermark: float = 0.35,
    high_watermark: float = 0.55,
    overwrite: bool = False,
) -> dict[str, str]:
    """Configure PyTorch MPS allocator watermarks to prevent macOS SSD swap stalls on 16-32 GB Macs.

    Must be called before the first `torch.device("mps")` allocation to constrain the Metal heap pool.
    """
    if not (0.0 < low_watermark <= high_watermark <= 1.0):
        raise ValueError(f"Expected 0 < low_watermark <= high_watermark <= 1, got ({low_watermark}, {high_watermark})")

    low_s = f"{low_watermark:.2f}"
    high_s = f"{high_watermark:.2f}"
    if overwrite or "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = low_s
    if overwrite or "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high_s
    return {
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"],
    }


def mps_allocated_gib() -> float:
    """Return active tensor memory allocated by PyTorch on MPS in GiB (`0.0` if unavailable)."""
    if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return float(torch.mps.current_allocated_memory()) / float(1024**3)
        except Exception:
            return 0.0
    return 0.0


def mps_driver_gib() -> float:
    """Return total Metal driver heap memory reserved by the process in GiB (`0.0` if unavailable)."""
    if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
        try:
            return float(torch.mps.driver_allocated_memory()) / float(1024**3)
        except Exception:
            return 0.0
    return 0.0


def maybe_trim_mps_cache(step: int, every: int = 4, max_driver_gib: float | None = None) -> bool:
    """Periodically release unreferenced MPS cached pages so Unified Memory never drifts into swap."""
    if not (hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False
    should_trim = (every > 0 and step > 0 and (step % every == 0))
    if not should_trim and max_driver_gib is not None:
        should_trim = mps_driver_gib() >= max_driver_gib
    if should_trim:
        torch.mps.empty_cache()
        return True
    return False

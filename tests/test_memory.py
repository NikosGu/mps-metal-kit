"""Unit tests for Unified Memory watermark configuration and MPS cache helpers."""
from __future__ import annotations

import os
import pytest

import mps_metal_kit as mmk


def test_configure_mps_memory_sets_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    cfg = mmk.configure_mps_memory(0.35, 0.52)
    assert cfg["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.35"
    assert cfg["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.52"
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.52"


def test_configure_mps_memory_validates_bounds():
    with pytest.raises(ValueError):
        mmk.configure_mps_memory(0.70, 0.50)


def test_mps_memory_metrics_and_trim_safe():
    assert mmk.mps_allocated_gib() >= 0.0
    assert mmk.mps_driver_gib() >= 0.0
    res = mmk.maybe_trim_mps_cache(step=4, every=4)
    assert isinstance(res, bool)

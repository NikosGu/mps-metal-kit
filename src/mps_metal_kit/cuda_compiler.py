"""Runtime JIT compiler and loader for the cross-platform C++20 / CUDA / FP64 extension."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import threading
import torch

_CUDA_EXT_MOD = None
_CUDA_EXT_LOCK = threading.Lock()
_PKG_DIR = Path(__file__).resolve().parent

_CUDA_SOURCES = (
    "cuda_bindings.cpp",
    "chunk_scan_cuda.cu",
    "token_scan_cuda.cu",
    "spectral_cuda.cu",
    "ragged_cuda.cu",
)
_CUDA_HEADERS = ("cuda_ops.hpp",)


def is_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def load_cuda_extension(force_rebuild: bool = False):
    """Compile (if not cached) and load `mps_metal_kit_cuda_ext` for NVIDIA GPUs."""
    global _CUDA_EXT_MOD
    if _CUDA_EXT_MOD is not None and not force_rebuild:
        return _CUDA_EXT_MOD

    if not is_cuda_available():
        raise RuntimeError("CUDA backend requires a CUDA-capable GPU and PyTorch CUDA build.")

    with _CUDA_EXT_LOCK:
        if _CUDA_EXT_MOD is not None and not force_rebuild:
            return _CUDA_EXT_MOD

        from torch.utils.cpp_extension import load

        csrc_dir = _PKG_DIR / "cuda_csrc"
        hasher = hashlib.sha256()
        for h in _CUDA_HEADERS:
            hasher.update((csrc_dir / h).read_bytes())
        src_paths = []
        for s in _CUDA_SOURCES:
            p = csrc_dir / s
            src_paths.append(str(p))
            hasher.update(p.read_bytes())

        py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
        hasher.update(f"{torch.__version__}:{torch.version.cuda}:{py_tag}".encode("utf-8"))
        digest = hasher.hexdigest()[:14]

        cache_root = Path(
            os.environ.get("MPS_METAL_KIT_CACHE_DIR") or (Path(tempfile.gettempdir()) / "mps_metal_kit_cache")
        )
        build_dir = cache_root / f"cuda_{digest}"
        build_dir.mkdir(parents=True, exist_ok=True)

        _CUDA_EXT_MOD = load(
            name=f"mps_metal_kit_cuda_{digest}",
            sources=src_paths,
            extra_include_paths=[str(csrc_dir)],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=str(build_dir),
            verbose=False,
        )
        return _CUDA_EXT_MOD

"""Runtime JIT compiler and loader for the modular Objective-C++20 / Metal / Accelerate extension."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import threading
import torch

_EXT_MOD = None
_EXT_LOCK = threading.Lock()
_PKG_DIR = Path(__file__).resolve().parent

_HEADER_FILES = (
    "mps_bridge.hpp",
    "shader_registry.hpp",
    "amx_lapack.hpp",
    "ops.hpp",
)
_CPP_UNITS = (
    "bindings.mm",
    "scans_ops.mm",
    "spectral_ops.mm",
    "ragged_ops.mm",
)
_SHADER_FILES = (
    "chunk_scan.metal",
    "token_scan.metal",
    "spectral.metal",
    "ragged.metal",
)


def is_mps_available() -> bool:
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def header_include_dir() -> str:
    """Return the directory containing `mps_metal_kit/*.hpp` for external C++/Metal extensions."""
    return str(_PKG_DIR / "include")


def load_extension(force_rebuild: bool = False):
    """Compile (if not cached) and load `mps_metal_kit_ext`, registering all modular Metal shaders."""
    global _EXT_MOD
    if _EXT_MOD is not None and not force_rebuild:
        return _EXT_MOD

    if not is_mps_available():
        raise RuntimeError("mps-metal-kit requires macOS with an Apple Silicon MPS-capable GPU.")

    with _EXT_LOCK:
        if _EXT_MOD is not None and not force_rebuild:
            return _EXT_MOD

        from torch.utils.cpp_extension import include_paths, library_paths

        inc_dir = _PKG_DIR / "include" / "mps_metal_kit"
        csrc_dir = _PKG_DIR / "csrc"
        shd_dir = _PKG_DIR / "shaders"

        hasher = hashlib.sha256()
        for name in _HEADER_FILES:
            hasher.update((inc_dir / name).read_bytes())
        mm_paths = []
        for name in _CPP_UNITS:
            p = csrc_dir / name
            mm_paths.append(str(p))
            hasher.update(p.read_bytes())
        shader_sources = []
        for name in _SHADER_FILES:
            src_txt = (shd_dir / name).read_text(encoding="utf-8")
            shader_sources.append(src_txt)
            hasher.update(src_txt.encode("utf-8"))

        py_tag = f"{sys.version_info.major}.{sys.version_info.minor}"
        hasher.update(f"{torch.__version__}:{py_tag}".encode("utf-8"))
        digest = hasher.hexdigest()[:16]

        cache_root = Path(
            os.environ.get("MPS_METAL_KIT_CACHE_DIR") or (Path(tempfile.gettempdir()) / "mps_metal_kit_cache")
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        so_path = cache_root / f"mps_metal_kit_ext_{digest}.so"

        if force_rebuild or not so_path.exists():
            tmp_so = cache_root / f"mps_metal_kit_ext_{digest}.{os.getpid()}.so"
            py_inc = sysconfig.get_paths()["include"]
            inc_flags = [f"-I{header_include_dir()}", f"-I{py_inc}"] + [f"-I{p}" for p in include_paths()]
            lib_flags = []
            for p in library_paths():
                lib_flags.extend([f"-L{p}", f"-Wl,-rpath,{p}"])

            clang_bin = os.environ.get("CXX", "/usr/bin/clang++")
            cmd = [
                clang_bin,
                "-ObjC++",
                "-std=c++20",
                "-O3",
                "-shared",
                "-fPIC",
                "-undefined",
                "dynamic_lookup",
                "-DTORCH_EXTENSION_NAME=mps_metal_kit_ext",
                *inc_flags,
                *lib_flags,
                "-ltorch",
                "-ltorch_cpu",
                "-lc10",
                "-ltorch_python",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-framework",
                "Accelerate",
                *mm_paths,
                "-o",
                str(tmp_so),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Failed to compile mps_metal_kit_ext (exit {proc.returncode}):\n{proc.stderr}"
                )
            os.replace(tmp_so, so_path)

        spec = importlib.util.spec_from_file_location("mps_metal_kit_ext", str(so_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not create import spec for {so_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.init_shader_modules(*shader_sources)
        _EXT_MOD = mod
        return _EXT_MOD

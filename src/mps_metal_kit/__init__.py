"""mps-metal-kit: C++20 / Metal / Accelerate & CUDA / FP64 kernel toolkit for PyTorch MPS and CUDA."""

from .compiler import header_include_dir, is_mps_available, load_extension
from .cuda_compiler import is_cuda_available, load_cuda_extension
from .memory import (
    configure_mps_memory,
    maybe_trim_mps_cache,
    mps_allocated_gib,
    mps_driver_gib,
)
from .ragged import ragged_waves_linear, reference_ragged_waves_linear
from .scans import (
    chunked_delta_rule,
    deltanet_chunk_scan,
    reference_deltanet_chunk_scan,
    reference_token_delta_scan,
    token_delta_scan,
)
from .spectral import (
    reference_eigh,
    reference_weighted_gram_subspace,
    unified_eigh,
    weighted_gram_subspace,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "is_mps_available",
    "is_cuda_available",
    "load_extension",
    "load_cuda_extension",
    "header_include_dir",
    "configure_mps_memory",
    "mps_allocated_gib",
    "mps_driver_gib",
    "maybe_trim_mps_cache",
    "deltanet_chunk_scan",
    "reference_deltanet_chunk_scan",
    "token_delta_scan",
    "reference_token_delta_scan",
    "chunked_delta_rule",
    "unified_eigh",
    "reference_eigh",
    "weighted_gram_subspace",
    "reference_weighted_gram_subspace",
    "ragged_waves_linear",
    "reference_ragged_waves_linear",
]

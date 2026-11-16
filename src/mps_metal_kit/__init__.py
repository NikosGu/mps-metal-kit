"""mps-metal-kit: Zero-copy C++20 / Objective-C++ / Metal / Accelerate toolkit for PyTorch MPS."""

from .compiler import header_include_dir, is_mps_available, load_extension
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "is_mps_available",
    "load_extension",
    "header_include_dir",
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

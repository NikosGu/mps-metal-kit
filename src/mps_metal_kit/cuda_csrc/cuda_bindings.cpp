#include "cuda_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mps-metal-kit CUDA C++20 / FP64 extension";
    m.def("deltanet_chunk_scan_fwd", &mps_kit::cuda_ops::deltanet_chunk_scan_fwd);
    m.def("deltanet_chunk_scan_bwd", &mps_kit::cuda_ops::deltanet_chunk_scan_bwd);
    m.def("token_delta_scan_fwd", &mps_kit::cuda_ops::token_delta_scan_fwd);
    m.def("token_delta_scan_bwd", &mps_kit::cuda_ops::token_delta_scan_bwd);
    m.def("cuda_eigh_fwd", &mps_kit::cuda_ops::cuda_eigh_fwd);
    m.def("cuda_dual_gram_subspace_fwd", &mps_kit::cuda_ops::cuda_dual_gram_subspace_fwd);
    m.def("spectral_subspace_bwd", &mps_kit::cuda_ops::spectral_subspace_bwd);
    m.def("ragged_waves_mm_fwd", &mps_kit::cuda_ops::ragged_waves_mm_fwd);
    m.def("ragged_waves_mm_bwd", &mps_kit::cuda_ops::ragged_waves_mm_bwd);
}

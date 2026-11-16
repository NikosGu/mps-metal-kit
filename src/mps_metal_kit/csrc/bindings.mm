#include "mps_metal_kit/ops.hpp"
#include "mps_metal_kit/shader_registry.hpp"

namespace {

void init_shader_modules(
    const std::string& chunk_src,
    const std::string& token_src,
    const std::string& spectral_src,
    const std::string& ragged_src)
{
    auto& reg = mps_kit::PipelineRegistry::instance();
    reg.register_library("chunk_scan", chunk_src, true);
    reg.register_library("token_scan_64", "#define DK_TOK 64\n" + token_src, true);
    reg.register_library("token_scan_128", "#define DK_TOK 128\n" + token_src, true);
    reg.register_library("spectral", spectral_src, true);
    reg.register_library("ragged", ragged_src, true);
}

} // namespace

PYBIND11_MODULE(mps_metal_kit_ext, m) {
    m.doc() = "Zero-copy C++20 / Objective-C++ / Metal / Accelerate extension for PyTorch MPS";

    m.def("init_shader_modules", &init_shader_modules, "Compile and register modular Metal shader libraries");

    m.def("deltanet_chunk_scan_fwd", &mps_kit::ops::deltanet_chunk_scan_fwd, "Chunk matrix scan forward");
    m.def("deltanet_chunk_scan_bwd", &mps_kit::ops::deltanet_chunk_scan_bwd, "Chunk matrix scan backward");
    m.def("token_delta_scan_fwd", &mps_kit::ops::token_delta_scan_fwd, "Token gated delta scan forward");
    m.def("token_delta_scan_bwd", &mps_kit::ops::token_delta_scan_bwd, "Token gated delta scan backward");

    m.def("amx_eigh_fwd", &mps_kit::ops::amx_eigh_fwd, "Zero-copy AMX symmetric eigensolver");
    m.def("amx_dual_gram_subspace_fwd", &mps_kit::ops::amx_dual_gram_subspace_fwd, "Zero-copy AMX Gram subspace forward");
    m.def("spectral_subspace_bwd", &mps_kit::ops::spectral_subspace_bwd, "Metal spectral subspace backward");

    m.def("ragged_waves_mm_fwd", &mps_kit::ops::ragged_waves_mm_fwd, "Single-encoder ragged projection forward");
    m.def("ragged_waves_mm_bwd", &mps_kit::ops::ragged_waves_mm_bwd, "Single-encoder ragged projection backward");
}

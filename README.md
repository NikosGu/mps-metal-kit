# mps-metal-kit

**Zero-Copy C++20 / Objective-C++ / Metal / Accelerate Kernel Toolkit for PyTorch MPS on Apple Silicon.**

`mps-metal-kit` eliminates the Python dispatch, command-encoder splitting, dynamic `MPSGraph` padding, and CPU-GPU copy bottlenecks that limit custom sequence-modeling, linear-attention, spectral, and ragged-batch operators on PyTorch's `mps` backend.

---

## Why & Where to Use `mps-metal-kit`

On NVIDIA GPUs, researchers rely on **Triton** or **`flash-linear-attention`** to fuse recurrent state scans, ragged routing, and custom backward passes. On **Apple Silicon (`mps`)**, Triton is unavailable, `torch.linalg.eigh` / `float64` forces expensive CPU round-trips, and Python token loops stall the GPU with thousands of tiny command encoders.

`mps-metal-kit` fills that gap for four concrete workloads on Mac:

1. **Sub-Quadratic / Linear-Attention & State-Space Models on Mac** (`DeltaNet`, `Gated DeltaNet`, `RWKV`, Fast-Weight Programmers, Test-Time Training / `TTT` layers): train and run constant-memory $O(T)$ recurrent layers at **`36x` higher speed** without materializing `[B, H, T, D_v, D_k]` states in Unified Memory (`examples/gated_deltanet_attention.py`).
2. **Differentiable Spectral Layers, PCA Pooling, Whitening & Orthogonal / Riemannian Optimization**: run exact `float64` symmetric eigendecompositions (`unified_eigh`) and variable-length principal subspace extraction (`weighted_gram_subspace`) on Apple's **AMX coprocessor** at zero tensor-copy cost, with analytical Daleckii-Krein gradients computed directly on the Metal GPU (`examples/spectral_subspace_pooling.py`).
3. **Ragged / Unpadded Document Packing, Chunked Prefill & Sparse MoE Routing**: project variable-length segments (`ragged_waves_linear`) with `1` threadgroup per active token—eliminating `5x-20x` rectangular padding waste and fusing multi-wave dispatches into a single Metal command encoder.
4. **C++20 / Metal SDK for Custom PyTorch `mps` Extensions**: include `mps_metal_kit/mps_bridge.hpp` in your own `.mm` extension to bind PyTorch `mps` tensors directly to custom Metal compute shaders and `Accelerate` LAPACK/BLAS in ~20 lines of C++20.

---

## Highlights

- **Zero-Copy Unified Memory Bridge (`mps_bridge.hpp`)**: Accesses PyTorch `mps` tensors directly as raw `id<MTLBuffer>` handles for Metal compute shaders *and* as coherent host pointers (`[MTLBuffer contents]`) for Apple Silicon's **AMX coprocessor (`Accelerate`)**—with **zero `.cpu()` / `.to("mps")` copies**.
- **Single-Encoder `MPSStream` Loop Fusion (`encode_on_torch_stream`)**: Encodes multi-step or multi-wave kernel loops directly onto PyTorch's active `id<MTLComputeCommandEncoder>` in a single C++20 pass (`~20 us` CPU overhead, zero command-buffer splits, zero host sync stalls).
- **Register-Resident Fast-Weight & DeltaNet Scans**:
  - `token_delta_scan`: Token-level gated DeltaNet / fast-weight recurrence holding the `[64 x D_k]` state matrix in hardware registers across the sequence with `CKI=16` checkpointed exact backpropagation and zero-barrier `simd_sum` transpose reductions (**`36x` faster** than PyTorch MPS).
  - `deltanet_chunk_scan` & `chunked_delta_rule`: WY-factorized chunkwise causal DeltaNet and general matrix-transition scan $S_{n+1} = \gamma_n M_n S_n + B_n$ with exact gradients $(\nabla M, \nabla B, \nabla \gamma, \nabla S_0)$.
- **Hybrid Apple AMX `float64` Forward + Metal GPU Spectral Backward**:
  - `unified_eigh`: Differentiable symmetric eigendecomposition (`dsyevd_` with `dsyev_` fallback) reading and writing MPS Unified Memory in place.
  - `weighted_gram_subspace`: Automatic **Dual Gram** ($L \times L$ when $L < D$) vs **Primal Gram** ($D \times D$ when $L \ge D$) top-$r$ orthonormal subspace extraction on AMX paired with a fused $O(M \cdot r \cdot D)$ Daleckii-Krein backward kernel on the Metal GPU (**`14x` faster** than PyTorch).
- **Prefix-Sum Ragged Multi-Wave Projections (`ragged_waves_linear`)**: Launches `1` threadgroup per active token via binary search on prefix-sum offsets—eliminating rectangular padding amplification across variable-length segments.

---

## Requirements & Installation

- **OS**: macOS 13.0+ on Apple Silicon (`M1` / `M2` / `M3` / `M4`)
- **Compiler**: Xcode Command Line Tools (`/usr/bin/clang++` with `-ObjC++ -std=c++20`)
- **Python**: `>= 3.10` with `torch >= 2.2.0`

```bash
git clone https://github.com/NikosGu/mps-metal-kit.git
cd mps-metal-kit
pip install -e ".[dev]"
```

The native Objective-C++20 / Metal / Accelerate shared library is compiled automatically on first use and cached by content SHA-256 in `$TMPDIR/mps_metal_kit_cache` (configurable via `MPS_METAL_KIT_CACHE_DIR`).

---

## Quickstart

### 1. Register-Resident Gated DeltaNet / Fast-Weight Scan

```python
import torch
import torch.nn.functional as F
import mps_metal_kit as mmk

dev = torch.device("mps")
B, H, T, DK, DV = 4, 8, 512, 128, 64

q = F.normalize(torch.randn(B, H, T, DK, device=dev), dim=-1).requires_grad_(True)
k = F.normalize(torch.randn(B, H, T, DK, device=dev), dim=-1).requires_grad_(True)
v = torch.randn(B, H, T, DV, device=dev, requires_grad=True)
beta = torch.sigmoid(torch.randn(B, H, T, device=dev) - 1.0).requires_grad_(True)
gamma = torch.sigmoid(torch.randn(B, H, T, device=dev) + 2.0).requires_grad_(True)

# Exact register-resident Metal forward + CKI=16 checkpointed backward
out, final_state = mmk.token_delta_scan(q, k, v, beta, gamma)
out.sum().backward()
```

For WY-factorized chunk-parallel causal DeltaNet (`gamma = 1`):

```python
out, final_state = mmk.chunked_delta_rule(q, k, v, beta, chunk_size=64)
```

### 2. Zero-Copy AMX `float64` Eigensolver & Subspace Extraction on MPS

```python
import torch
import mps_metal_kit as mmk

dev = torch.device("mps")

# Differentiable symmetric eigendecomposition directly on MPS tensors
A = torch.randn(8, 64, 64, device=dev)
G = (A @ A.transpose(-1, -2)).requires_grad_(True)
eigvals, eigvecs = mmk.unified_eigh(G)

# Top-r orthonormal subspace of G_b = sum_{t < L_b} w_{b,t} x_{b,t} x_{b,t}^T
# Forward: zero-copy Unified-Memory AMX (Dual Gram when L_b < D, Primal Gram when L_b >= D)
# Backward: fused single-encoder Metal GPU Daleckii-Krein kernel
X = torch.randn(16, 32, 256, device=dev, requires_grad=True)
weights = torch.rand(16, 32, device=dev, requires_grad=True)
U = mmk.weighted_gram_subspace(X, weights, rank=4, lengths=[24] * 16)
projector = U @ U.transpose(-1, -2)
projector.pow(2).sum().backward()
```

### 3. Single-Encoder Multi-Wave Ragged Projections

```python
import torch
import mps_metal_kit as mmk

dev = torch.device("mps")
x = torch.randn(4, 256, 128, device=dev, requires_grad=True)
wave_weights = [torch.randn(4, 128, 128, device=dev, requires_grad=True) for _ in range(8)]
wave_intervals = [
    [(b, w * 16, w * 16 + 12) for b in range(4)]
    for w in range(8)
]

# Encodes all 8 ragged waves into ONE Metal compute command encoder
y = mmk.ragged_waves_linear(x, wave_weights, wave_intervals)
y.sum().backward()
```

### 4. Writing Your Own C++20 / Metal Kernels with `mps_bridge.hpp`

External extensions can include `mps_metal_kit/mps_bridge.hpp` directly:

```python
import mps_metal_kit as mmk
print(mmk.header_include_dir())
```

```cpp
#include <mps_metal_kit/mps_bridge.hpp>

void launch_custom_kernel(torch::Tensor input, torch::Tensor output) {
    auto b_in = mps_kit::UnifiedBuffer::from_tensor(input);
    auto b_out = mps_kit::UnifiedBuffer::from_tensor(output);
    auto pso = mps_kit::PipelineRegistry::instance().pipeline("my_lib", "my_kernel");

    mps_kit::encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        b_in.bind(enc, 0);
        b_out.bind(enc, 1);
        [enc dispatchThreadgroups:MTLSizeMake(input.size(0), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}
```

---

## Benchmarks (Apple Silicon M2 Pro)

End-to-end **Forward + Backward** latency measured via `python benchmarks/bench_mps_metal_kit.py`:

| Operator (`Forward + Backward`) | PyTorch `mps` | `mps-metal-kit` | Speedup |
| :--- | ---: | ---: | ---: |
| `token_delta_scan` `[B=4, H=8, T=512, 64x128]` | `446.63 ms` | `12.41 ms` | **`35.98x`** |
| `weighted_gram_subspace` `[B=16, M=32, D=192, r=8]` | `34.22 ms` | `2.45 ms` | **`13.97x`** |
| `ragged_waves_linear` `[B=16, 32 waves, D=192]` | `120.33 ms` | `41.88 ms` | **`2.87x`** |
| `deltanet_chunk_scan` `[B=8, H=8, N=32, 64x128]` | `146.63 ms` | `68.58 ms` | **`2.14x`** |

All kernels are verified against exact `float64` / PyTorch autograd references (`pytest -v`, max forward error `< 2e-5`, max gradient error `< 8e-5`).

---

## Repository Layout

```text
mps-metal-kit/
├── src/mps_metal_kit/
│   ├── include/mps_metal_kit/
│   │   ├── mps_bridge.hpp        # Zero-copy UnifiedBuffer & MPSStream single-encoder scope
│   │   ├── shader_registry.hpp   # Thread-safe MSL runtime compiler & pipeline cache
│   │   ├── amx_lapack.hpp        # Apple AMX Accelerate float64 symmetric eigensolver
│   │   └── ops.hpp               # C++20 operator dispatch declarations
│   ├── shaders/
│   │   ├── chunk_scan.metal      # Register-resident chunk matrix-state scan
│   │   ├── token_scan.metal      # Register-resident token gated DeltaNet scan (CKI=16)
│   │   ├── spectral.metal        # Fused Daleckii-Krein spectral subspace backward
│   │   └── ragged.metal          # Prefix-sum active-token ragged linear kernels
│   ├── csrc/
│   │   ├── bindings.mm           # Pybind11 module entry point & shader registration
│   │   ├── scans_ops.mm          # Scan C++20 dispatchers
│   │   ├── spectral_ops.mm       # Zero-copy AMX eigh & Dual/Primal Gram C++20 solvers
│   │   └── ragged_ops.mm         # Single-encoder multi-wave ragged dispatchers
│   ├── compiler.py               # Content-hashed multi-unit Objective-C++20 JIT loader
│   ├── scans.py                  # Autograd wrappers & pure PyTorch references for scans
│   ├── spectral.py               # Autograd wrappers & pure PyTorch references for spectral ops
│   └── ragged.py                 # Autograd wrapper & pure PyTorch reference for ragged ops
├── docs/                         # API_REFERENCE.md, ARCHITECTURE.md, BENCHMARKS.md
├── examples/                     # gated_deltanet_attention.py, spectral_subspace_pooling.py
├── benchmarks/                   # bench_mps_metal_kit.py
└── tests/                        # test_scans.py, test_spectral.py, test_ragged.py
```

---

## License

Released under the [MIT License](LICENSE).

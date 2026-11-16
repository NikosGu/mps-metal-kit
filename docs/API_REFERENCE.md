# API Reference

Complete reference for both the **Python (`torch.autograd`) API** and the **C++20 / Objective-C++ Header SDK (`mps_bridge.hpp`)**.

---

## 1. Runtime & Extension Helpers (`mps_metal_kit.compiler`)

### `is_mps_available() -> bool`
Returns `True` when running on macOS (`darwin`) with an active Apple Silicon Metal Performance Shaders (`mps`) device.

### `load_extension(force_rebuild: bool = False) -> ModuleType`
Compiles (on first invocation) and loads the native Objective-C++20 / Metal / Accelerate shared library, registering all Metal compute pipelines (`MTLComputePipelineState`).
- **Environment variables**:
  - `MPS_METAL_KIT_CACHE_DIR`: Custom directory for cached `.so` binaries (defaults to `$TMPDIR/mps_metal_kit_cache`).
  - `CXX`: Override C++ compiler path (defaults to `/usr/bin/clang++`).

### `header_include_dir() -> str`
Returns the directory path containing `mps_metal_kit/mps_bridge.hpp` for `-I` compiler flags in external C++/Metal extensions.

---

## 2. Associative & Linear-Attention Scans (`mps_metal_kit.scans`)

### `token_delta_scan(q, k, v, beta, gamma=None, S0=None) -> tuple[Tensor, Tensor]`
Register-resident token-by-token gated fast-weight / DeltaNet recurrence with `CKI = 16` periodic state checkpointing:

$$e_t = v_t - S_{t-1} k_t, \qquad S_t = \gamma_t S_{t-1} + \beta_t e_t k_t^\top, \qquad o_t = S_t q_t$$

- **Parameters**:
  - `q` (`Tensor[B, H, T, D_k]`): Query vectors (`float32` or `bfloat16`). Metal fast path supports `D_k in {64, 128}`; falls back automatically to `reference_token_delta_scan` for other sizes or CPU.
  - `k` (`Tensor[B, H, T, D_k]`): Key / address vectors (typically $\ell_2$-normalized so $\beta_t \in (0, 2)$ ensures contraction).
  - `v` (`Tensor[B, H, T, D_v]`): Value vectors (`D_v = 64` on Metal fast path).
  - `beta` (`Tensor[B, H, T]`): Per-token scalar write rate $\beta_t$.
  - `gamma` (`Tensor[B, H, T] | None`): Optional scalar retention/forget gate $\gamma_t \in [0, 1]$ (defaults to `1.0`).
  - `S0` (`Tensor[B, H, D_v, D_k] | None`): Optional initial fast-weight state (defaults to zeros).
- **Returns**:
  - `out` (`Tensor[B, H, T, D_v]`): Sequence readout $o_0 \dots o_{T-1}$.
  - `S_end` (`Tensor[B, H, D_v, D_k]`): Terminal state $S_T$ (for recurrent chunk/streaming continuation).
- **Autograd**: Exact analytical gradients for all 6 inputs `(q, k, v, beta, gamma, S0)` without materializing `[B, H, T, D_v, D_k]` in DRAM.

---

### `deltanet_chunk_scan(M, B_in, decay=None, S0=None) -> tuple[Tensor, Tensor]`
Register-resident chunk-level affine matrix state recurrence:

$$S_{n+1} = \gamma_n \, (M_n S_n) + B_n$$

- **Parameters**:
  - `M` (`Tensor[B, H, N, D_k, D_k]`): Per-chunk transition matrices (`1 <= D_k <= 64` on Metal fast path, including non-powers of two).
  - `B_in` (`Tensor[B, H, N, D_k, D_v]`): Per-chunk additive state writes (`1 <= D_v <= 128`).
  - `decay` (`Tensor[B, H, N] | None`): Optional per-chunk scalar decay $\gamma_n$ (defaults to `1.0`).
  - `S0` (`Tensor[B, H, D_k, D_v] | None`): Optional initial state (defaults to zeros).
- **Returns**:
  - `St_hist` (`Tensor[B, H, N, D_k, D_v]`): Pre-chunk states $[S_0, S_1, \dots, S_{N-1}]$ needed for parallel intra-chunk readout.
  - `St_end` (`Tensor[B, H, D_k, D_v]`): Final state $S_N$.
- **Autograd**: Exact gradients for `(M, B_in, decay, S0)`.

---

### `chunked_delta_rule(q, k, v, beta, chunk_size=64, S0=None) -> tuple[Tensor, Tensor]`
End-to-end WY-factorized chunk-parallel causal DeltaNet layer operator combining intra-chunk batched triangular solves/GEMMs with `deltanet_chunk_scan` across chunks.
- **Parameters**:
  - `q, k` (`Tensor[B, H, T, D_k]`), `v` (`Tensor[B, H, T, D_v]`), `beta` (`Tensor[B, H, T]`), `chunk_size` (`int`, default `64`), `S0` (`Tensor[B, H, D_k, D_v] | None`).
- **Returns**: `(out [B, H, T, D_v], S_end [B, H, D_k, D_v])`.

---

## 3. Zero-Copy AMX Spectral Operators (`mps_metal_kit.spectral`)

### `unified_eigh(G) -> tuple[Tensor, Tensor]`
Differentiable symmetric eigendecomposition for PyTorch `mps` tensors—solving a major PyTorch MPS gap (`torch.linalg.eigh` is natively unsupported on MPS in many PyTorch builds or forces explicit CPU copies).
- **Parameters**:
  - `G` (`Tensor[..., D, D]`): Symmetric (or symmetrized $\frac{1}{2}(G + G^\top)$) tensor on `mps` (`float32` or `bfloat16`).
- **Returns**:
  - `L` (`Tensor[..., D]`): Ascending eigenvalues.
  - `V` (`Tensor[..., D, D]`): Orthonormal eigenvectors (`G @ V = V @ diag(L)`).
- **Autograd**: Exact Lorentzian / Daleckii-Krein backward on MPS.

---

### `weighted_gram_subspace(X, weights, rank=4, lengths=None, rel_tol=1e-5, min_eig=1e-6) -> Tensor`
Extracts the top-`rank` principal orthonormal basis $U_b \in \mathbb{R}^{D \times r}$ of the weighted empirical second-moment / Gram matrix:

$$G_b = \sum_{t=0}^{L_b - 1} \max(w_{b,t}, 0)\, x_{b,t} x_{b,t}^\top$$

- **Parameters**:
  - `X` (`Tensor[B, M, D]`): Token/feature vectors (`D <= 256` on Metal backward fast path).
  - `weights` (`Tensor[B, M]`): Non-negative importance weights per vector.
  - `rank` (`int`, default `4`): Number of top eigenvectors to retain (`rank <= 12` on Metal fast path).
  - `lengths` (`list[int] | Tensor[B] | None`): True unpadded length $L_b \le M$ per batch element.
  - `rel_tol` (`float`, default `1e-5`): Relative eigenvalue cutoff ($\lambda_i > \text{rel\_tol} \cdot \lambda_{\max}$).
  - `min_eig` (`float`, default `1e-6`): Minimum leading eigenvalue threshold below which $U_b = 0$.
- **Returns**:
  - `U` (`Tensor[B, D, rank]`): Orthonormal columns for kept modes, zero columns for inactive/dropped modes.
- **Important**: Eigenvector signs $\pm u_a$ are arbitrary; downstream objectives should consume the sign-invariant orthogonal projector $P = U U^\top$.

---

## 4. Ragged Multi-Wave Operators (`mps_metal_kit.ragged`)

### `ragged_waves_linear(x, wave_weights, wave_intervals) -> Tensor`
Applies variable-length per-row linear projections across multiple sequential/ragged waves inside **one** Metal compute command encoder, launching `1` threadgroup per active token (zero wasted threadgroups on inactive positions).
- **Parameters**:
  - `x` (`Tensor[B, T, D_in]`): Input sequence (`D_in <= 256`).
  - `wave_weights` (`list[Tensor[B, D_out, D_in]]`): Weight matrix per wave (`D_out <= 256`).
  - `wave_intervals` (`list[list[tuple[int, int, int]]]`): Active `(row_b, start_t, stop_t)` slices per wave.
- **Returns**: `Y` (`Tensor[B, T, D_out]`).

---

## 5. C++20 / Objective-C++ Header SDK (`mps_metal_kit/mps_bridge.hpp`)

| Symbol | Purpose |
| :--- | :--- |
| `mps_kit::UnifiedBuffer::from_tensor(t)` | Extracts `id<MTLBuffer>`, byte offset, and coherent CPU pointer (`data_as<T>()`) from any contiguous `mps` `torch::Tensor` with zero copies. |
| `mps_kit::PipelineRegistry::instance()` | Compiles MSL source strings at runtime (`register_library`) and caches `id<MTLComputePipelineState>` objects (`pipeline`). |
| `mps_kit::encode_on_torch_stream(fn)` | Borrows PyTorch's active `MPSStream` `id<MTLComputeCommandEncoder>` inside `dispatch_sync(stream->queue(), ...)` so an entire C++ loop of kernel dispatches runs in one encoder without command-buffer splits. |
| `mps_kit::sync_torch_stream()` | Commits and waits on PyTorch's `MPSStream` before zero-copy CPU/AMX reads or writes on `[MTLBuffer contents]`. |
| `mps_kit::amx_sym_eigh_f64(n, A, w)` | Symmetric `float64` eigensolver via Apple `Accelerate` (`dsyevd_` with automatic workspace query and `dsyev_` fallback). |

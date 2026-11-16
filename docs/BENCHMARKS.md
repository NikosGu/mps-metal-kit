# Benchmarks & Numerical Verification

## Reproducing Benchmarks

On an Apple Silicon Mac with Xcode Command Line Tools installed:

```bash
pip install -e ".[dev]"
pytest -v
python benchmarks/bench_mps_metal_kit.py
```

For quick CI validation with fewer iterations:

```bash
python benchmarks/bench_mps_metal_kit.py --quick
```

---

## Reference Results (Apple M2 Pro, Unified Memory, FP32 Forward + Backward)

| Workload | Shape / Config | Standard PyTorch `mps` | `mps-metal-kit` | Speedup | Primary Bottleneck Eliminated |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `token_delta_scan` | `B=4, H=8, T=512, DK=128, DV=64` | `446.63 ms` | `12.41 ms` | **`35.98x`** | `512x` Python loop launches + `[B, H, T, 64, 128]` HBM state traffic |
| `weighted_gram_subspace` | `B=16, M=32, D=192, rank=8` | `34.22 ms` | `2.45 ms` | **`13.97x`** | `.cpu()` / `.to("mps")` copies + `192x192` Primal Gram replaced by `32x32` AMX Dual Gram + Metal backward |
| `ragged_waves_linear` | `B=16, T=512, 32 waves, D=192` | `120.33 ms` | `41.88 ms` | **`2.87x`** | Per-segment `MPSGraph` command-encoder splits + tensor cloning |
| `deltanet_chunk_scan` | `B=8, H=8, N=32, DK=64, DV=128` | `146.63 ms` | `68.58 ms` | **`2.14x`** | `32x` sequential `matmul` dispatches & intermediate state materialization |

---

## Numerical Precision Guarantees

Every custom kernel in `mps-metal-kit` is tested in `tests/test_mps_metal_kit.py` against exact PyTorch / `float64` autograd references:

1. **`deltanet_chunk_scan`**: Tested across power-of-two (`64x128`, `32x64`) and non-power-of-two (`48x96`) state shapes; forward max error `< 2e-5`, all four input gradients (`dM, dB, d_decay, dS0`) max error `< 5e-5`.
2. **`token_delta_scan`**: Tested across `DK in {64, 128}` and sequences crossing multiple `CKI=16` checkpoint boundaries (`T=73`) with non-zero initial state `S0` and terminal state gradient `dS_T`; forward max error `< 2e-5`, all six gradients (`dQ, dK, dV, d_beta, d_gamma, dS0`) max error `< 5e-5`.
3. **`chunked_delta_rule`**: Verified end-to-end against token-by-token `token_delta_scan` (`max_err < 3e-5`).
4. **`unified_eigh` & `weighted_gram_subspace`**: Verified on gauge-invariant spectral objectives in both Dual-Gram ($M < D$) and Primal-Gram ($M \ge D$) regimes, including rank-deficient / zero-weight edge cases (`max_err < 5e-5` on `weighted_gram_subspace`).
5. **`ragged_waves_linear`**: Verified across 12 irregular variable-length waves encoded in a single Metal command encoder (`max_err < 2e-5` forward, `< 8e-5` backward).

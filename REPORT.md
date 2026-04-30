# Execution Report — Toy Model X-ray Pulse Profile

**Date:** April 30, 2026
**Repo:** `Toy-model_X-ray_pulse-profile`

---

## Summary

All core deliverables from the plan (`plan.md`) have been implemented. The project now contains a working toy forward model for X-ray pulse profile generation with three acceleration strategies: pure NumPy, Numba JIT, and JAX JIT. All three agree within floating-point precision.

## Files Created

| File | Description |
|------|-------------|
| `toymodel_numpy.py` | Baseline pure NumPy implementation — uses Beloborodov (2002) light-bending approximation, spot grid integration, blackbody flux |
| `toymodel_numba.py` | Numba `@njit` accelerated version — inner phase loop compiled to machine code via LLVM |
| `toymodel_jax.py` | JAX `@jit` accelerated version — GPU-compatible, same code runs on GPU by changing imports only |
| `benchmark.py` | Consistency verification + timing comparison + PNG plot generation |
| `benchmark.png` | Benchmark plot: pulse profile curve + bar chart of timing |
| `README.md` | Project documentation with setup, usage, physics, and parameter tables |
| `AGENTS.md` | Agent instruction file for future OpenCode sessions |
| `REPORT.md` | This file — execution report with errors and next steps |

## Benchmark Results

Test configuration: M=1.4 M☉, R=12 km, i=60°, θ=30°, ρ=10°, T=0.3 keV, 128 phases, 20×40 spot grid.

| Implementation | Time per call | vs NumPy |
|---------------|---------------|----------|
| NumPy         | 4.11 ms       | 1.0×     |
| Numba @njit   | 1.34 ms       | 3.1×     |
| JAX @jit      | 2.34 ms       | 1.8×     |

Verification: max absolute difference between any pair = 3.1e-15 (machine precision).

## Error Encountered & Fix

**Issue:** `llvmlite` (Numba dependency) fails to build from source on macOS because it requires LLVM/CMake which is not installed as a system library.

**Fix applied:** Install Numba via conda-forge instead of pip:
```bash
conda install -c conda-forge numba -y
```
This provides pre-built `llvmlite` binaries. The correction has been documented in `AGENTS.md`.

## Next Steps

1. **Push to GitHub** — this is the immediate next action (see below)
2. **ML Extension (optional)** — train an `sklearn.neural_network.MLPRegressor` emulator:
   - Input: (compactness, inclination, spot colatitude)
   - Output: pulse profile amplitude
   - Create a Jupyter notebook demonstrating the concept
3. **Increase spot resolution** — bump `n_spot_radial`/`n_spot_azimuthal` to 50×100 to show larger speedups (Numba/JAX advantage grows with grid size)
4. **Add comparison parameters** — benchmark across a grid of (M, R, i, θ) configurations to generate ML training data
5. **Link from personal website** — make the repo a visible portfolio piece for the DENSeR interview

## GitHub Remote

Target: `https://github.com/abubakryagob/Toy-model_X-ray_pulse-profile`

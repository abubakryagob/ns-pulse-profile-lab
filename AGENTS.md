# AGENTS.md — Toy Model X-ray Pulse Profile

## Project purpose

Build a minimal forward model for X-ray pulse-profile generation from a spinning neutron star with a single circular hot spot, as prep for the DENSeR postdoc interview. This is a **demonstration project** — accuracy matters less than showing you understand the X-PSI pipeline and can write accelerated scientific Python.

## Environment setup

llvmlite (numba dependency) fails to build from source on macOS without
LLVM/CMake. Install numba via conda-forge for pre-built binaries, then pip
the rest.

```bash
conda create -n denserprep python=3.11 -y
conda activate denserprep
conda install -c conda-forge numba -y
pip install numpy scipy matplotlib "jax[cpu]" cython jupyter scikit-learn
```

## Running the benchmark

```bash
conda activate denserprep
python benchmark.py
```

Generates `benchmark.png` with the pulse profile plot and timing comparison.

## Physics reference

Use the **Beloborodov (2002)** approximation for light bending:

```
1 - cos(α) ≈ (1 - cos(ψ)) × (1 - r_s/r)
```

where:
- `α` = emission angle from surface normal
- `ψ` = angle between emission point and observer (from star's centre)
- `r_s = 2GM/c²` = Schwarzschild radius
- Compactness `u = r_s / R`

A spot is visible when `cos(α) > 0`, i.e. `α < 90°`. If `inclination + colatitude < 90°`, the spot is never occulted; if `> 90°`, it partially disappears behind the star.

## Toy model deliverables

Three implementations of the same forward model, benchmarked against each other:

1. **Pure NumPy** — baseline
2. **Numba `@njit`** on the inner loop — demonstrates ~10–50× speedup
3. **JAX `@jit`** — same code runs on GPU by changing imports only

Inputs: `M`, `R`, `i` (observer inclination), `θ` (spot colatitude), `ρ` (spot angular radius), `T` (temperature), `D` (distance).

Output: phase-resolved flux curve (pulse profile).

## Optional extension

Use `sklearn.neural_network.MLPRegressor` to train a small emulator: input = (compactness, inclination, spot colatitude), output = pulse profile amplitude. Demonstrates the ML emulator concept from the DENSeR proposal.

## Repo conventions

- No existing code — all structure is to be created.
- The plan lives in `plan.md` at the repo root. Read it for full context.
- Target GitHub: `https://github.com/abubakryagob`
- Include a `README.md` explaining the project, a benchmark plot (PNG), and a Jupyter notebook if doing the ML extension.

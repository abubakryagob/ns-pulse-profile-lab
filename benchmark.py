"""Benchmark the three toy-model implementations and verify consistency."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import toymodel_numpy as np_impl
import toymodel_numba as nb_impl
import toymodel_jax as jx_impl
import jax.numpy as jnp


# Default neutron star parameters
PARAMS = {
    "M": 1.4,         # Solar masses
    "R": 12.0,        # km
    "i_deg": 60.0,    # observer inclination
    "theta_deg": 30.0,  # spot colatitude
    "rho_deg": 10.0,  # spot angular radius
    "T_keV": 0.3,     # spot temperature
    "D_kpc": 1.0,     # distance
    "n_spot_radial": 20,
    "n_spot_azimuthal": 40,
}

N_PHASES = 128
N_WARMUP = 3
N_TIMING = 20


def run_benchmark():
    phases = np.linspace(0, 1, N_PHASES)

    print("=" * 60)
    print("X-ray Pulse Profile Forward Model — Benchmark")
    print("=" * 60)
    print(f"Parameters: M={PARAMS['M']} Msun, R={PARAMS['R']} km, "
          f"i={PARAMS['i_deg']}°, θ={PARAMS['theta_deg']}°, "
          f"ρ={PARAMS['rho_deg']}°")
    print(f"Resolution: {N_PHASES} phases, "
          f"{PARAMS['n_spot_radial']}x{PARAMS['n_spot_azimuthal']} spot grid")
    print()

    # --- NumPy ---
    print("1. Pure NumPy...")
    flux_np = np_impl.pulse_profile(phases, **PARAMS)
    for _ in range(N_WARMUP):
        np_impl.pulse_profile(phases, **PARAMS)
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        np_impl.pulse_profile(phases, **PARAMS)
    t_np = (time.perf_counter() - t0) / N_TIMING
    print(f"   Time: {t_np*1000:.2f} ms per call")

    # --- Numba ---
    print("2. Numba @njit...")
    # First call triggers compilation
    flux_nb = nb_impl.pulse_profile(phases, **PARAMS)
    for _ in range(N_WARMUP):
        nb_impl.pulse_profile(phases, **PARAMS)
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        nb_impl.pulse_profile(phases, **PARAMS)
    t_nb = (time.perf_counter() - t0) / N_TIMING
    print(f"   Time: {t_nb*1000:.2f} ms per call")

    # --- JAX ---
    print("3. JAX @jit...")
    phases_jax = jnp.array(phases)
    flux_jx = jx_impl.pulse_profile(phases_jax, **PARAMS)
    # JAX JIT triggers on first call; warm up
    for _ in range(N_WARMUP):
        flux_jx = jx_impl.pulse_profile(phases_jax, **PARAMS)
    flux_jx.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_TIMING):
        flux_jx = jx_impl.pulse_profile(phases_jax, **PARAMS)
        flux_jx.block_until_ready()
    t_jx = (time.perf_counter() - t0) / N_TIMING
    print(f"   Time: {t_jx*1000:.2f} ms per call")

    # --- Verification ---
    print()
    print("--- Verification ---")
    flux_jx_np = np.asarray(flux_jx)
    diff_nb = np.max(np.abs(flux_np - flux_nb))
    diff_jx = np.max(np.abs(flux_np - flux_jx_np))
    print(f"Max |NumPy - Numba| = {diff_nb:.2e}")
    print(f"Max |NumPy - JAX|   = {diff_jx:.2e}")

    # --- Speedup ---
    speedup_nb = t_np / t_nb
    speedup_jx = t_np / t_jx
    print()
    print("--- Speedup ---")
    print(f"Numba vs NumPy: {speedup_nb:.1f}x")
    print(f"JAX vs NumPy:   {speedup_jx:.1f}x")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Pulse profile
    ax1.plot(phases, flux_np * 1e13, "k-", linewidth=2, label="NumPy")
    ax1.plot(phases, flux_nb * 1e13, "r--", linewidth=1.5, label="Numba")
    ax1.plot(phases, flux_jx_np * 1e13, "b:", linewidth=1.5, label="JAX")
    ax1.set_xlabel("Rotational Phase")
    ax1.set_ylabel("Flux [10$^{-13}$ erg/cm$^2$/s]")
    ax1.set_title(f"Pulse Profile  (M={PARAMS['M']} M$_\\odot$, "
                  f"R={PARAMS['R']} km, i={PARAMS['i_deg']}°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Benchmark bar chart
    times_ms = [t_np * 1000, t_nb * 1000, t_jx * 1000]
    labels = ["NumPy", "Numba", "JAX"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax2.bar(labels, times_ms, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Time per call [ms]")
    ax2.set_title(f"Benchmark ({N_PHASES} phases, "
                  f"{PARAMS['n_spot_radial']}x{PARAMS['n_spot_azimuthal']} grid)")
    for bar, tval in zip(bars, times_ms):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{tval:.1f} ms", ha="center", va="bottom", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=150)
    print()
    print("Benchmark plot saved to benchmark.png")

    return flux_np, flux_nb, flux_jx_np


if __name__ == "__main__":
    run_benchmark()

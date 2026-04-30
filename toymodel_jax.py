"""JAX-accelerated forward model for X-ray pulse profiles.

Same physics as toymodel_numpy.py but using JAX @jit for GPU-compatible
vectorised computation. Runs on CPU or GPU by changing jax import only.

Reference: Beloborodov A.M., 2002, ApJL, 566, L85
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap

# Physical constants (CGS)
G = 6.67430e-8
c = 2.99792458e10
M_sun = 1.98847e33
sigma_sb = 5.670374419e-5
keV_to_K = 1.160451812e7


def schwarzschild_radius(M_solar):
    return 2.0 * G * M_solar * M_sun / c**2


def compactness(M_solar, R_km):
    r_s = schwarzschild_radius(M_solar)
    return r_s / (R_km * 1e5)


def _build_grids(rho, n_spot_radial, n_spot_azimuthal):
    """Build integration grids (not jit-compiled, called once)."""
    delta_edges = jnp.linspace(0.0, rho, n_spot_radial + 1)
    gammas = jnp.linspace(0.0, 2.0 * jnp.pi, n_spot_azimuthal + 1)[:-1]
    dgamma = 2.0 * jnp.pi / n_spot_azimuthal

    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    ddelta = delta_edges[1:] - delta_edges[:-1]

    weights = jnp.sin(delta_centers) * ddelta * dgamma

    delta_grid = delta_centers[:, None] + jnp.zeros((1, n_spot_azimuthal))
    gamma_grid = gammas[None, :] + jnp.zeros((n_spot_radial, 1))
    weights_2d = weights[:, None] + jnp.zeros((1, n_spot_azimuthal))

    return delta_grid, gamma_grid, weights_2d


@jit
def _cos_theta_p(cos_theta, sin_theta, delta_grid, gamma_grid):
    return (cos_theta * jnp.cos(delta_grid)
            - sin_theta * jnp.sin(delta_grid) * jnp.cos(gamma_grid))


@jit
def _dphi(cos_theta, sin_theta, delta_grid, gamma_grid, cos_theta_p):
    sin_theta_p = jnp.sqrt(jnp.clip(1.0 - cos_theta_p**2, 0.0, 1.0))
    sin_delta = jnp.sin(delta_grid)

    sin_dphi = jnp.where(sin_theta_p > 1e-12,
                         sin_delta * jnp.sin(gamma_grid) / sin_theta_p,
                         0.0)
    cos_dphi = jnp.where(sin_theta_p > 1e-12,
                         (jnp.cos(delta_grid) - cos_theta_p * cos_theta)
                         / (sin_theta_p * sin_theta),
                         1.0)

    return jnp.arctan2(jnp.clip(sin_dphi, -1.0, 1.0),
                       jnp.clip(cos_dphi, -1.0, 1.0))


@jit
def _compute_one_phase(phi_obs, cos_theta_p, sin_theta_p, dphi,
                       weights_2d, cos_i, sin_i, inv_u):
    """Compute flux for a single rotational phase."""
    cos_psi = jnp.clip(
        cos_i * cos_theta_p
        + sin_i * sin_theta_p * jnp.cos(dphi - phi_obs),
        -1.0, 1.0)

    cos_alpha = 1.0 - (1.0 - cos_psi) * inv_u
    contrib = jnp.where(cos_alpha > 0.0, cos_alpha * weights_2d, 0.0)
    return jnp.sum(contrib)


# vmap over phases: vectorised compute across all phases at once
_compute_all_phases = jit(vmap(_compute_one_phase, in_axes=(0, None, None, None, None, None, None, None)))


def pulse_profile(phases, M, R, i_deg, theta_deg, rho_deg, T_keV, D_kpc,
                  n_spot_radial=20, n_spot_azimuthal=40):
    """Compute phase-resolved flux (erg/cm^2/s) at each rotational phase.

    Same interface as toymodel_numpy.pulse_profile.
    """
    i = jnp.radians(i_deg)
    theta = jnp.radians(theta_deg)
    rho = jnp.radians(rho_deg)
    u = compactness(M, R)
    T = T_keV * keV_to_K
    D = D_kpc * 3.085677581e21
    R_cm = R * 1e5

    I_surf = sigma_sb * T**4 / jnp.pi

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Build grids once (not inside jit)
    delta_grid, gamma_grid, weights_2d = _build_grids(
        rho, n_spot_radial, n_spot_azimuthal)

    cos_theta_p = _cos_theta_p(cos_theta, sin_theta, delta_grid, gamma_grid)
    sin_theta_p = jnp.sqrt(jnp.clip(1.0 - cos_theta_p**2, 0.0, 1.0))
    dphi_array = _dphi(cos_theta, sin_theta, delta_grid, gamma_grid, cos_theta_p)

    cos_i = jnp.cos(i)
    sin_i = jnp.sin(i)
    inv_u = 1.0 - u

    prefactor = I_surf * R_cm**2 / D**2
    redshift_factor = jnp.sqrt(1.0 - u)

    phi_obs = 2.0 * jnp.pi * phases

    flux_per_phase = _compute_all_phases(
        phi_obs, cos_theta_p, sin_theta_p, dphi_array,
        weights_2d, cos_i, sin_i, inv_u)

    return prefactor * (redshift_factor**3) * flux_per_phase

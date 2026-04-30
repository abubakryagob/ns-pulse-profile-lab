"""Numba-accelerated forward model for X-ray pulse profiles.

Same physics as toymodel_numpy.py but with @njit on the inner
phase-integration loop for 10-50x speedup.

Reference: Beloborodov A.M., 2002, ApJL, 566, L85
"""

import numpy as np
import numba

# Physical constants (CGS)
G = 6.67430e-8
c = 2.99792458e10
M_sun = 1.98847e33
sigma_sb = 5.670374419e-5
keV_to_K = 1.160451812e7


@numba.njit(cache=True)
def _cos_theta_p_grid(cos_theta, sin_theta, delta_grid, gamma_grid):
    """Compute cos(theta_p) for spot grid points (star spherical coords)."""
    nr, na = delta_grid.shape
    result = np.empty_like(delta_grid)
    for ir in range(nr):
        for ia in range(na):
            result[ir, ia] = (cos_theta * np.cos(delta_grid[ir, ia])
                              - sin_theta * np.sin(delta_grid[ir, ia])
                              * np.cos(gamma_grid[ir, ia]))
    return result


@numba.njit(cache=True)
def _dphi_grid(cos_theta, sin_theta, delta_grid, gamma_grid, cos_theta_p):
    """Azimuthal offset dphi for each spot grid point."""
    nr, na = delta_grid.shape
    dphi = np.empty_like(delta_grid)
    for ir in range(nr):
        for ia in range(na):
            stp = np.sqrt(max(0.0, 1.0 - cos_theta_p[ir, ia]**2))
            if stp > 1e-12:
                ctp = cos_theta_p[ir, ia]
                sg = gamma_grid[ir, ia]
                cd = delta_grid[ir, ia]
                sd = np.sin(cd)
                sin_d = sd * np.sin(sg) / stp
                cos_d = (np.cos(cd) - ctp * cos_theta) / (stp * sin_theta)
                dphi[ir, ia] = np.arctan2(max(-1.0, min(1.0, sin_d)),
                                          max(-1.0, min(1.0, cos_d)))
            else:
                dphi[ir, ia] = 0.0
    return dphi


@numba.njit(cache=True)
def pulse_profile(phases, M, R, i_deg, theta_deg, rho_deg, T_keV, D_kpc,
                  n_spot_radial=20, n_spot_azimuthal=40):
    """Compute phase-resolved flux (erg/cm^2/s) at each rotational phase.

    Same interface as toymodel_numpy.pulse_profile.
    """
    # Compactness
    r_s = 2.0 * G * M * M_sun / c**2
    R_cm = R * 1e5
    u = r_s / R_cm

    i = np.radians(i_deg)
    theta = np.radians(theta_deg)
    rho = np.radians(rho_deg)
    T = T_keV * keV_to_K
    D = D_kpc * 3.085677581e21

    I_surf = sigma_sb * T**4 / np.pi

    n_phases = len(phases)

    delta_edges = np.linspace(0.0, rho, n_spot_radial + 1)
    gammas = np.linspace(0.0, 2.0 * np.pi, n_spot_azimuthal + 1)[:-1]
    dgamma = 2.0 * np.pi / n_spot_azimuthal

    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    ddelta = delta_edges[1:] - delta_edges[:-1]

    weights = np.sin(delta_centers) * ddelta * dgamma

    delta_grid = np.empty((n_spot_radial, n_spot_azimuthal))
    gamma_grid = np.empty((n_spot_radial, n_spot_azimuthal))
    weights_2d = np.empty((n_spot_radial, n_spot_azimuthal))
    for ir in range(n_spot_radial):
        for ia in range(n_spot_azimuthal):
            delta_grid[ir, ia] = delta_centers[ir]
            gamma_grid[ir, ia] = gammas[ia]
            weights_2d[ir, ia] = weights[ir]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cos_theta_p = _cos_theta_p_grid(cos_theta, sin_theta, delta_grid, gamma_grid)
    dphi = _dphi_grid(cos_theta, sin_theta, delta_grid, gamma_grid, cos_theta_p)
    sin_theta_p = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_p**2))

    cos_i = np.cos(i)
    sin_i = np.sin(i)

    prefactor = I_surf * R_cm**2 / D**2
    redshift_factor = np.sqrt(1.0 - u)
    inv_u = 1.0 - u

    flux = np.empty(n_phases)

    for k in range(n_phases):
        phi_obs = 2.0 * np.pi * phases[k]
        total = 0.0

        for ir in range(n_spot_radial):
            for ia in range(n_spot_azimuthal):
                cos_psi = (cos_i * cos_theta_p[ir, ia]
                           + sin_i * sin_theta_p[ir, ia]
                           * np.cos(dphi[ir, ia] - phi_obs))
                if cos_psi > 1.0:
                    cos_psi = 1.0
                elif cos_psi < -1.0:
                    cos_psi = -1.0

                cos_alpha = 1.0 - (1.0 - cos_psi) * inv_u
                if cos_alpha > 0.0:
                    total += cos_alpha * weights_2d[ir, ia]

        flux[k] = prefactor * (redshift_factor**3) * total

    return flux

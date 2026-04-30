"""Pure NumPy forward model for X-ray pulse profiles from a neutron star
with a single circular hot spot. Uses the Beloborodov (2002) light-bending
approximation.

Reference: Beloborodov A.M., 2002, ApJL, 566, L85
"""

import numpy as np

# Physical constants (CGS)
G = 6.67430e-8        # cm^3 / g / s^2
c = 2.99792458e10     # cm / s
M_sun = 1.98847e33    # g
sigma_sb = 5.670374419e-5  # erg / cm^2 / s / K^4
keV_to_K = 1.160451812e7  # K / keV


def schwarzschild_radius(M_solar):
    """Schwarzschild radius in cm for mass in solar masses."""
    return 2.0 * G * M_solar * M_sun / c**2


def compactness(M_solar, R_km):
    """Compactness u = r_s / R."""
    r_s = schwarzschild_radius(M_solar)
    return r_s / (R_km * 1e5)


def spot_visible(psi_cos, u):
    """Check if the spot centre is visible using Beloborodov (2002).

    cos(alpha) = 1 - (1 - cos(psi)) * (1 - u)
    Spot visible when cos(alpha) > 0  =>  alpha < 90 deg.
    """
    cos_alpha = 1.0 - (1.0 - psi_cos) * (1.0 - u)
    return cos_alpha > 0.0, cos_alpha


def pulse_profile(phases, M, R, i_deg, theta_deg, rho_deg, T_keV, D_kpc,
                  n_spot_radial=20, n_spot_azimuthal=40):
    """Compute phase-resolved flux from a spinning NS with a single hot spot.

    Parameters
    ----------
    phases : array_like
        Rotational phases (0 to 1, where 0 is the spot directly facing observer).
    M : float
        Neutron star mass in solar masses.
    R : float
        Neutron star radius in km.
    i_deg : float
        Observer inclination in degrees (0 = pole-on).
    theta_deg : float
        Spot colatitude in degrees (0 = rotation pole).
    rho_deg : float
        Spot angular radius in degrees.
    T_keV : float
        Spot temperature in keV (blackbody).
    D_kpc : float
        Distance to source in kpc.
    n_spot_radial : int
        Number of radial samples within the spot.
    n_spot_azimuthal : int
        Number of azimuthal samples within the spot.

    Returns
    -------
    flux : ndarray
        Observed flux (erg/cm^2/s) at each rotational phase.
    """
    phases = np.asarray(phases, dtype=np.float64)

    i = np.radians(i_deg)
    theta = np.radians(theta_deg)
    rho = np.radians(rho_deg)
    u = compactness(M, R)
    T = T_keV * keV_to_K
    D = D_kpc * 3.085677581e21  # kpc -> cm
    R_cm = R * 1e5               # km -> cm

    # Surface brightness (erg / cm^2 / s / sterad)
    I_surf = sigma_sb * T**4 / np.pi

    # Spot integration points in local polar coords (δ, γ)
    # δ = angular distance from spot centre, γ = azimuth
    deltas = np.linspace(0, rho, n_spot_radial)
    gammas = np.linspace(0, 2 * np.pi, n_spot_azimuthal + 1)[:-1]

    # Build 2D grids
    delta_grid, gamma_grid = np.meshgrid(deltas, gammas, indexing='ij')

    # Solid angle weight per element (sin δ dδ dγ)
    # Approximate integration using midpoints
    delta_edges = np.linspace(0, rho, n_spot_radial + 1)
    dgamma = 2 * np.pi / n_spot_azimuthal

    # Weight for each element: area ≈ sin(δ) * Δδ * Δγ
    delta_centers = 0.5 * (delta_edges[:-1] + delta_edges[1:])
    ddelta = delta_edges[1:] - delta_edges[:-1]

    weights = np.sin(delta_centers) * ddelta * dgamma
    weights_2d = weights[:, np.newaxis] * np.ones(len(gammas))

    # Spherical law of cosines to compute angle ψ between
    # each spot point and the observer
    # Point on star: colatitude θ_p, phase φ_p
    # Observer direction: colatitude i, phase φ_obs
    # cos(ψ) = cos(i)cos(θ_p) + sin(i)sin(θ_p)cos(φ_p - φ_obs)

    # Transform spot-local coords to star spherical coords
    cos_theta_p = (np.cos(theta) * np.cos(delta_grid)
                   - np.sin(theta) * np.sin(delta_grid) * np.cos(gamma_grid))
    sin_theta_p = np.sqrt(np.clip(1.0 - cos_theta_p**2, 0, 1))
    sin_delta = np.sin(delta_grid)

    # Azimuthal offset from spot centre in star frame
    # sin(Δφ) = sin(δ)sin(γ)/sin(θ_p) when θ_p not at pole
    sin_dphi = np.where(sin_theta_p > 1e-12,
                        sin_delta * np.sin(gamma_grid) / sin_theta_p,
                        0.0)
    cos_dphi = np.where(sin_theta_p > 1e-12,
                        (np.cos(delta_grid) - cos_theta_p * np.cos(theta))
                        / (sin_theta_p * np.sin(theta)),
                        1.0)
    dphi = np.arctan2(np.clip(sin_dphi, -1, 1), np.clip(cos_dphi, -1, 1))

    flux = np.empty(len(phases))
    n_elements = n_spot_radial * n_spot_azimuthal

    # Pre-factor
    prefactor = I_surf * R_cm**2 / D**2

    for k, phase in enumerate(phases):
        phi_obs = 2.0 * np.pi * phase

        # Angle between each spot point and observer
        cos_psi = (np.cos(i) * cos_theta_p
                   + np.sin(i) * sin_theta_p * np.cos(dphi - phi_obs))
        cos_psi = np.clip(cos_psi, -1.0, 1.0)

        # Beloborodov bending
        cos_alpha = 1.0 - (1.0 - cos_psi) * (1.0 - u)
        visible = cos_alpha > 0.0

        # Flux: surface brightness * area * projection * redshift * visibility
        # Gravitational redshift: (1+z) = 1/sqrt(1-u)
        redshift_factor = np.sqrt(1.0 - u)

        # Observed specific intensity: I_obs = I_emit / (1+z)^3
        # Integration: F = ∫ I_emit * cos(alpha) * dΩ_spot * (R/D)^2 / (1+z)^3
        # Actually: dF = I_emit * cos(alpha) * d(area) / D^2 * 1/(1+z)^3
        # where d(area) = R^2 * sin(δ) dδ dγ

        contrib = np.where(visible, cos_alpha * weights_2d, 0.0)
        flux[k] = prefactor * redshift_factor**3 * np.sum(contrib)

    return flux

"""Chromatic shear as a SHARED Kalman state (2026-11-11): two pyramids on either side of the science wavelength (e.g. 450 and 750 nm for
600 nm) see the science phase plus dn_i c, with dn_i the refractivity difference to the science wavelength and c the shear per unit dn of
that mode. KalmanFilterData.from_ar2_closed_loop(common_*) puts c in the state, seen with gain dn_i by sensor i: the filter estimates the
SCIENCE phase (the shear cancelled to first order, the noise weighted).

Model of the e2e propagation (AtmoPropagation.compute_chromatic_shifts): n - 1 of MatharAirRefraction at standard conditions (15 C,
1013.25 hPa, 0 %), the shift of layer l per unit dn  b_l = sec z tan z (P0 / (g rho_s)) (1 - P(h_l) / P0) (Devaney 2024, NASA profile),
the monochromatic wavelength of each source. Per mode j at k_j = (n_j + 1) / 2D, small shifts:
    var(c_j) = var(phi_j) * (2 pi k_j)^2 / 2 * sum_l cn2_l b_l^2               (per unit dn^2; isotropic average of (k.b)^2)
c in units of DN_UNIT (gains dn_i / DN_UNIT). var(phi_j): the stationary variance of the filter's AR(2) (same units as the filter).
Dynamics: the AR(2) of every mode with its natural frequency scaled by v_c / v_phi, v_c the 5/3 moment of the winds weighted by
cn2 b^2 (the high layers), v_phi the cn2-weighted one (the wind of the turbulence AR(2)); same damping.
"""
import os.path as op

import numpy as np
import yaml

from specula.data_objects.kalman_filter_data import ar2_driving_noise_variance, ar2_stationary_variance
from specula.lib.air_refraction import MatharAirRefraction
from specula.processing_objects.atmo_propagation import AtmoPropagation

E2E_DIR = op.dirname(op.abspath(__file__))
DN_UNIT = 1e-6
G, RHO_S = 9.8, 1.225


def _yml(main_yml='ristretto_main.yml'):
    with open(op.join(E2E_DIR, main_yml)) as f:
        return yaml.safe_load(f)


def refractivity_std(wvl_nm):
    """n - 1 at the standard conditions of the e2e propagation."""
    return float(MatharAirRefraction().get_refractive_index(float(wvl_nm) * 1e-9))


def layer_shifts_per_dn(heights, zenith_deg, altitude_m):
    """[m] lateral shift of every layer per unit dn (compute_chromatic_shifts / delta_N)."""
    z = np.radians(zenith_deg)
    P0 = AtmoPropagation._pressure_nasa(altitude_m)
    b0 = (1 / np.cos(z)) * np.tan(z) * (P0 * 100.0) / (G * RHO_S)
    return np.array([b0 * (1.0 - AtmoPropagation._pressure_nasa(altitude_m + float(h)) / P0) for h in heights])


def gains(wvls_nm, sci_nm):
    """dn_i / DN_UNIT of every sensor wavelength."""
    n_sci = refractivity_std(sci_nm)
    return [(refractivity_std(w) - n_sci) / DN_UNIT for w in wvls_nm]


def interpolation_weights(wvls_nm, sci_nm):
    """Weights of two sensors with sum w_i dn_i = 0 and sum w_i = 1 (the noise-free limit of the filter)."""
    g1, g2 = gains(wvls_nm, sci_nm)
    w1 = -g2 / (g1 - g2)
    return w1, 1.0 - w1


def common_block(a1, a2, q, wvls_nm, sci_nm, diameter, zeta, fs, main_yml='ristretto_main.yml'):
    """dict(var, a1, a2, gain) for save_kalman_data(common=...): a1, a2, q the turbulence AR(2) of the filter at the loop rate fs
    (n_modes,), zeta its damping (sweep_contrast.ZETA)."""
    from aoeb import utils as au
    y = _yml(main_yml)
    heights, cn2 = np.asarray(y['atmo']['heights'], float), np.asarray(y['atmo']['Cn2'], float)
    cn2 = cn2 / cn2.sum()
    wind = np.asarray(y['wind_speed']['constant'], float)
    b = layer_shifts_per_dn(heights, y['main']['zenithAngleInDeg'], y['prop'].get('telescope_altitude_m', 0.0)) * DN_UNIT
    n_modes = len(a1)
    k = (au.radial_order(np.arange(n_modes)) + 1.0) / (2.0 * diameter)
    var = ar2_stationary_variance(a1, a2, q) * (2 * np.pi * k) ** 2 / 2 * float(np.sum(cn2 * b ** 2))
    w_c = cn2 * b ** 2
    v_c = (np.sum(w_c * wind ** (5 / 3)) / w_c.sum()) ** (3 / 5)
    v_phi = (np.sum(cn2 * wind ** (5 / 3))) ** (3 / 5)
    dt = 1.0 / fs
    omega = -np.log(np.sqrt(-np.asarray(a2, float))) / (zeta * dt) * (v_c / v_phi)
    r = np.exp(-zeta * omega * dt)
    ca1, ca2 = 2 * r * np.cos(omega * dt * np.sqrt(1 - zeta ** 2)), -r ** 2
    return dict(var=var, a1=ca1, a2=ca2, gain=gains(wvls_nm, sci_nm), v_c=float(v_c), v_phi=float(v_phi))

"""Per-mode AR(2) model for the Kalman filter, from noiseless open-loop modal series.

    x[k] = a1 x[k-1] + a2 x[k-2] + w[k],  var(w) = q

Only two measured quantities per mode are used: var(x) and var(dx). For frozen flow on axis,
var(x) does not depend on the layers, and var(dx) scales as sum(cn2 * wind**2).
"""
import numpy as np


def wind_rms(cn2, wind):
    """Cn2-weighted rms wind speed: the only profile number the model depends on."""
    cn2, wind = np.asarray(cn2, float), np.asarray(wind, float)
    return np.sqrt(np.sum(cn2 * wind ** 2) / np.sum(cn2))


def add_moments(moments, atmo_modes):
    """Accumulate one run, shaped (n_steps, n_modes). The mean is not removed on purpose."""
    atmo_modes = np.asarray(atmo_modes, dtype=float)
    var = np.mean(atmo_modes ** 2, axis=0)
    dvar = np.mean(np.diff(atmo_modes, axis=0) ** 2, axis=0)
    return (var, dvar, 1) if moments is None else (moments[0] + var, moments[1] + dvar, moments[2] + 1)


def ar2_model(moments, dt, wind_scale=1.0, var_scale=1.0, zeta=0.7, var_degree=2):
    """Returns a1, a2, q per mode (mode rank = index + 1), smoothed over the mode rank.

    wind_scale = wind_rms(new profile) / wind_rms(measured profile)
    var_scale  = (r0 measured / r0 new)**(5/3)
    """
    var, dvar, n_runs = moments
    var, dvar = var / n_runs, dvar / n_runs
    log_j = np.log(np.arange(var.size) + 1.0)
    weights = np.exp(-log_j / 2)                 # same total weight per decade of j

    omega = np.sqrt(dvar / var) / dt
    omega_fit = np.exp(np.polyval(np.polyfit(log_j, np.log(omega), 1, w=weights), log_j)) * wind_scale
    var_fit = np.exp(np.polyval(np.polyfit(log_j, np.log(var), var_degree, w=weights), log_j)) * var_scale

    r = np.exp(-zeta * omega_fit * dt)
    a1 = 2 * r * np.cos(omega_fit * dt * np.sqrt(1 - zeta ** 2))
    a2 = -r ** 2
    q = var_fit * (1 + a2) * ((1 - a2) ** 2 - a1 ** 2) / (1 - a2)
    return a1, a2, q

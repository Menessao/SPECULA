"""
Helper functions shared by the main_simul, run_simul and plots notebooks of this folder.

The calibration helpers read their files from the module-level ``calib_dir``: to use another
directory, set ``utils.calib_dir = '...'`` (rebinding the name imported in a notebook is not enough).
"""
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

E2E_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of the YAML files

calib_dir = '/raid1/mmenessini/calibration/WFSW'
result_dir = '/raid1/mmenessini/results/WFSW'


############################ Running SPECULA ############################

def run_specula(*yml_files, log_file='specula.log', cwd=E2E_DIR, check=True, extra_args=()):
    """Run ``specula <yml_files>`` in a subprocess (replaces os.system('specula ...')).

    stdout and stderr go to ``log_file`` (relative to ``cwd``), overwritten at every call.
    Raises subprocess.CalledProcessError if the simulation fails and check is True.
    """
    yml_files = [f if f.endswith(('.yml', '.yaml')) else f + '.yml' for f in yml_files]
    log_path = os.path.join(cwd, log_file)
    with open(log_path, 'w') as log:
        result = subprocess.run(['specula', *extra_args, *yml_files], cwd=cwd,
                                stdout=log, stderr=subprocess.STDOUT)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args,
                                            output=f'see {log_path}')
    return result.returncode


def latest_tn(store_dir=None):
    """Most recent tracking-number folder (YYYYMMDD_HHMMSS) written by the DataStore."""
    store_dir = result_dir if store_dir is None else store_dir
    tns = [d for d in os.listdir(store_dir)
           if os.path.isdir(os.path.join(store_dir, d)) and d[:8].isdigit() and d[8:9] == '_']
    return sorted(tns)[-1]


############################ Reconstructors (main_simul) ############################

def rec_mmse(im, mean_frame, pup, ron_var, rcond=1e-12, use_cross=True, flux=None):
    """R = (D^T Cn^-1 D)^-1 D^T Cn^-1, with im = D of shape (2N, nmodes)."""
    f = np.asarray(mean_frame, dtype=np.float64).ravel()
    if flux is not None:
        f = f * flux / f.sum()
    A, B, C, Dp = [f[pup[:, k][pup[:, k] >= 0]] for k in range(4)]
    F = (A + B + C + Dp).mean()
    ac, bd = np.maximum(A + C, 0), np.maximum(B + Dp, 0)
    var_s = (ac + bd + 4 * ron_var) / F**2        # Cn diagonal
    cov_xy = (bd - ac) / F**2 if use_cross else np.zeros_like(var_s)
    det = var_s**2 - cov_xy**2
    inv_d, inv_o = (var_s / det)[:, None], (-cov_xy / det)[:, None]
    N = var_s.size
    D = im
    Cinv_D = np.vstack([inv_d * D[:N] + inv_o * D[N:],
                        inv_o * D[:N] + inv_d * D[N:]])
    C_modes = np.linalg.pinv(D.T @ Cinv_D, rcond=rcond, hermitian=True)
    return C_modes @ Cinv_D.T, C_modes


def plot_rec_cov(rec,nSubap:int=48):
    plt.loglog(np.arange(rec.shape[0])+1,np.diag(rec @ rec.T)*(nSubap**2)/(48**2))


############################ Noise propagation (plots, run_simul) ############################

X = np.array([1, 1, -1, -1])    # x^k for pupils A, B, C, D
Y = np.array([-1, 1, 1, -1])    # y^k for pupils A, B, C, D

def rec_noise_var(rec, mean_frame, pup, sigma_e, excess=1.0):
    """Per-mode photon and read-noise variance, in rec units squared."""
    f = mean_frame.ravel()
    I = np.array([f[pup[:, k][pup[:, k] >= 0]] for k in range(4)])
    sum_I = I.sum(axis=0).mean()
    N = I.shape[1]
    Rx, Ry = rec[:, :N], rec[:, N:]
    phot = np.zeros(rec.shape[0])
    read = np.zeros(rec.shape[0])
    for k in range(4):
        w2 = (X[k] * Rx + Y[k] * Ry)**2
        phot += excess**2 * (w2 @ np.maximum(I[k], 0.0))
        read += w2.sum(axis=1) * sigma_e**2
    return phot / sum_I**2, read / sum_I**2

def noise_sensitivities(rec, mean_frame, pup, wvl=None):
    """s_phot, s_read, Npix. flux defaults to the sum over the whole frame."""
    flux = np.sum(mean_frame)
    phot, read = rec_noise_var(rec, mean_frame, pup, sigma_e=1.0)
    if wvl is None:
        wvl = 1.0
    s_phot = np.sqrt(phot * flux) / wvl
    s_read = np.sqrt(read) * flux / wvl
    return s_phot, s_read

def propagated_noise_var(s_phot, s_read, flux, sigma_e, wvl=1.0, excess=1.0):
    phot = (excess * s_phot * wvl)**2 / flux
    read = (s_read * sigma_e * wvl)**2 / flux**2
    return phot, read

def _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes=None):
    """Mean frame, pupil indices and reconstructor of one pyramid configuration."""
    nCorrModes = nModes if nCorrModes is None else nCorrModes
    tag = f'pyr{rMod:1.1f}_{nSubaps}x{nSubaps}'
    frame = fits.getdata(os.path.join(calib_dir, 'aliasing', f'{tag}_{nCorrModes}corrmodes_wvl{wvl}_frame.fits'))
    if frame.ndim == 3:                       # old files: stack of frames
        frame = np.mean(frame, axis=0)
    pup = fits.getdata(os.path.join(calib_dir, 'pupils', f'pyr_pupdata_{nSubaps}x{nSubaps}.fits'), 1)
    rec = fits.getdata(os.path.join(calib_dir, 'rec', f'{tag}_wvl{wvl}_bmc2k_vlt_{nModes}modes_rec.fits'), 1)
    return frame, pup, rec

def get_noise_propagation(rMod, nSubaps, nModes, wvl, sigma_e, flux, nCorrModes=None, pupil_flux=False):
    """Per-mode noise std, in reconstructor units (rec computed at wvl [nm]).
    flux: total frame flux [e-], or, with pupil_flux=True, the flux on the four pupils only
    (what PyrSlopec 'out_flux' reports)."""
    frame, pup, rec = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes)
    if pupil_flux:   # convert to total frame flux, the convention of noise_sensitivities
        pix = pup[pup >= 0]
        flux = flux * np.sum(frame) / np.sum(frame.ravel()[pix])
    s_phot, s_read = noise_sensitivities(rec, frame, pup, wvl=wvl)
    phot, read = propagated_noise_var(s_phot, s_read, flux=flux, sigma_e=sigma_e, wvl=wvl)
    return np.sqrt(phot+read)

def get_aliasing_power(rMod, nSubaps, nModes, wvl=900, nCorrModes=None, dt=5e-4):
    """Per-mode aliasing variance, single loop step (frames at 1/dt), in reconstructor units."""
    nCorrModes = nModes if nCorrModes is None else nCorrModes
    alias = fits.getdata(os.path.join(calib_dir, 'aliasing',
            f'pyr{rMod:1.1f}_{nSubaps}x{nSubaps}_{nCorrModes}corrmodes_{nModes}modes_wvl{wvl}_{1/dt:1.0f}Hz_alias.fits'))
    return alias if alias.ndim == 1 else np.var(alias, axis=0)   # files store the variance already

def get_atmo_coeffs(dt, nLayers=10, windRMS=12.9):
    data = fits.getdata(os.path.join(calib_dir,'atmo',f'turb_coeffs_{1/dt:1.0f}Hz_{nLayers}layers_{windRMS}wind.fits'))
    return data[0],data[1],data[2]


############################ Static offsets (run_simul) ############################

def pyr_slopes_from_frame(frame, pup):
    """Pyramid slopes [sx, sy] of a frame, as PyrSlopec computes them by default
    (no slope nulls, no threshold, global normalisation by the mean flux per sub-aperture)."""
    f = np.asarray(frame, dtype=np.float64).ravel()
    A, B, C, D = [f[pup[:, k][pup[:, k] >= 0]] for k in range(4)]
    norm = (A + B + C + D).sum() / pup.shape[0]
    return np.concatenate([A + B - C - D, B + C - A - D]) / norm

def get_static_modes(rMod, nSubaps, nModes, wvl, nCorrModes=None):
    """Modal measurement (reconstructor units) of the saved mean residual frame of one pyramid."""
    frame, pup, rec = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes)
    return rec @ pyr_slopes_from_frame(frame, pup)


def get_pupil_flux(rMod, nSubaps, nModes, wvl, nCorrModes=None):
    """Flux on the four pupils [e-] per loop step in the saved mean calibration frame
    (aliasing calibration of main_simul: cameras at the loop rate, magnitudes/QE of ristretto_main.yml).
    Same definition as PyrSlopec 'out_flux'."""
    frame, pup, _ = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes)
    return float(np.sum(frame.ravel()[pup[pup >= 0]]))

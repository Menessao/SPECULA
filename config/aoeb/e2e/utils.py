"""
Helper functions shared by the main_simul, run_simul and plots notebooks of this folder.

The calibration helpers read their files from the module-level ``calib_dir``: to use another
directory, set ``utils.calib_dir = '...'`` (rebinding the name imported in a notebook is not enough).
File names are those of aoeb (aoeb/naming.py, written by calibration.py): ``paths(wvl)`` gives them.
A sensing wavelength without products is derived from the nearest calibrated one (aoeb ASSUMPTIONS.md, A37):
its reconstructor is computed from the scaled IM and saved, the frames and slopes are those of the
calibrated wavelength at the equivalent seeing seeing * (wvl0 / wvl)^(6/5).
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

E2E_DIR = os.path.dirname(os.path.abspath(__file__))   # folder of the YAML files
sys.path.insert(0, os.path.dirname(os.path.dirname(E2E_DIR)))   # config/: the aoeb package
from aoeb.naming import Paths, nearest_wavelength, snap_seeing                       # noqa: E402
from aoeb.filter_optimizers import aliasing_frame_fraction                           # noqa: E402
from specula.data_objects.slopes import Slopes

import calibration                                                                    # noqa: E402

_CFG = calibration.E2EConfig()                         # aoeb/config/budget_wfsw.yml
calib_dir = os.environ.get('E2E_CALIB_DIR') or _CFG.calib_dir   # E2E_CALIB_DIR: another root for every importer (e.g. a staging root)
result_dir = '/raid1/mmenessini/results/WFSW'
PUPILSTOP_TAG, DM_TAG = _CFG.pupilstop_tag, _CFG.dm_tag
SEEING = 0.9          # seeing of the calibration products [arcsec] (the first e2e runs were made at 0.88 and are named s0.9)
D = 8.2               # [m]


def paths(wvl):
    """aoeb file layout of calib_dir at sensing wavelength wvl [nm]."""
    return Paths(calib_dir, PUPILSTOP_TAG, DM_TAG).at(float(wvl))


_MACHINES = {}


def machine():
    """aoeb budget machine of calib_dir (calibration.machine, cached per calib_dir)."""
    if calib_dir not in _MACHINES:
        _MACHINES[calib_dir] = calibration.machine(calibration.E2EConfig(calib_dir=calib_dir))
    return _MACHINES[calib_dir]


def _source(rMod, nSubaps, nCorrModes, wvl, seeing=SEEING):
    """(wvl0, seeing0): the products of wvl at that seeing, or those of the nearest wavelength with an average frame at the
    equivalent seeing seeing * (wvl0 / wvl)^(6/5) (rounded to 0.1 arcsec)."""
    seeing = snap_seeing(seeing)
    if os.path.exists(paths(wvl).avgframe(rMod, nSubaps, seeing, nCorrModes)):
        return float(wvl), seeing
    pat = os.path.join(calib_dir, 'frames', f'pyr{rMod:.1f}_{nSubaps}x{nSubaps}_wvl*_s*_{nCorrModes}modes_avgframe.fits')
    cands = sorted({float(re.search(r'_wvl([0-9]+)_', p).group(1)) for p in glob.glob(pat)} - {float(wvl)})
    if not cands:
        raise FileNotFoundError(f'no average frame for pyr{rMod:.1f}_{nSubaps}x{nSubaps}, {nCorrModes} modes corrected (calibration.py alias)')
    w0 = nearest_wavelength(float(wvl), cands)
    s0 = snap_seeing(seeing * (w0 / wvl) ** 1.2)
    if not os.path.exists(paths(w0).avgframe(rMod, nSubaps, s0, nCorrModes)):
        raise FileNotFoundError(f'{wvl:g} nm at {seeing} arcsec needs the {w0:g} nm products at {s0} arcsec (not simulated): calibrate '
                                'that seeing, or use the budget machine (utils.machine()), which interpolates in seeing')
    return w0, s0


def get_rec(rMod, nSubaps, nModes, wvl):
    """SVD reconstructor (nModes, n_slopes) at wvl [nm]; computed and saved from the scaled IM of the nearest wavelength if missing."""
    return np.asarray(fits.getdata(machine().ensure_rec(float(rMod), int(nSubaps), int(nModes), ('svd',), None, float(wvl))))


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


############################ Reconstructors ############################

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

def _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes=None, seeing=SEEING):
    """Mean frame, pupil indices and SVD reconstructor of one pyramid configuration at wvl [nm]."""
    nCorrModes = nModes if nCorrModes is None else nCorrModes
    w0, s0 = _source(rMod, nSubaps, nCorrModes, wvl, seeing)
    frame = fits.getdata(paths(w0).avgframe(rMod, nSubaps, s0, nCorrModes))
    pup = fits.getdata(paths(w0).pupdata(nSubaps), 1)
    return frame, pup, get_rec(rMod, nSubaps, nModes, wvl)

def get_noise_propagation(rMod, nSubaps, nModes, wvl, sigma_e, flux, nCorrModes=None, pupil_flux=False, seeing=SEEING):
    """Per-mode noise std, in reconstructor units (rec at wvl [nm]).
    flux: total frame flux [e-], or, with pupil_flux=True, the flux on the four pupils only
    (what PyrSlopec 'out_flux' reports)."""
    frame, pup, rec = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes, seeing)
    if pupil_flux:   # convert to total frame flux, the convention of noise_sensitivities
        pix = pup[pup >= 0]
        flux = flux * np.sum(frame) / np.sum(frame.ravel()[pix])
    s_phot, s_read = noise_sensitivities(rec, frame, pup)
    phot, read = propagated_noise_var(s_phot, s_read, flux=flux, sigma_e=sigma_e)
    return np.sqrt(phot+read)

def get_aliasing_power(rMod, nSubaps, nModes, wvl=900, nCorrModes=None, seeing=SEEING, fs=None):
    """Per-mode aliasing variance in reconstructor units (rec at wvl [nm]).
    fs=None: snapshot variance, as calibrated (independent screens every frame: no frame-rate dependence).
    fs [Hz]: variance of the aliasing averaged over ONE frame of a sensor at fs, the snapshot one times
    aoeb.filter_optimizers.aliasing_frame_fraction (AliasSpectrum, layers of ristretto_main.yml) -- the model of the budget machine.
    A wavelength without products: std(R_wvl @ slopes) of the nearest wavelength at the equivalent seeing (A37),
    rescaled to `seeing` as seeing^(5/3)."""
    nCorrModes = nModes if nCorrModes is None else nCorrModes
    w0, s0 = _source(rMod, nSubaps, nCorrModes, wvl, seeing)
    P0, k = paths(w0), (w0 / wvl) ** 1.2
    if w0 == wvl:
        alias = fits.getdata(P0.alias(rMod, nSubaps, s0, nModes, nCorrModes))
    elif os.path.exists(P0.slopes(rMod, nSubaps, s0, nCorrModes)):
        slopes = fits.getdata(P0.slopes(rMod, nSubaps, s0, nCorrModes)).astype(float)
        alias = np.std(get_rec(rMod, nSubaps, nModes, wvl) @ slopes.T, axis=1)
    else:
        alias = fits.getdata(P0.alias(rMod, nSubaps, s0, nModes, nCorrModes)) * (wvl / w0)
    var = np.asarray(alias, float) ** 2 * (float(seeing) * k / s0) ** (5 / 3)
    if fs is not None:
        var = var * aliasing_frame_fraction(fs, machine().alias_spectrum(wvl))
    return var

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

def get_static_modes(rMod, nSubaps, nModes, wvl, nCorrModes=None, seeing=SEEING):
    """Modal measurement (reconstructor units) of the saved mean residual frame of one pyramid."""
    frame, pup, rec = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes, seeing)
    return rec @ pyr_slopes_from_frame(frame, pup)

def get_static_slopes(rMod, nSubaps, nModes, wvl, nCorrModes=None, seeing=SEEING):
    frame, pup, _ = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes, seeing)
    return pyr_slopes_from_frame(frame, pup)

def get_pupil_flux(rMod, nSubaps, nModes, wvl, nCorrModes=None, seeing=SEEING):
    """Flux on the four pupils [e-] per loop step, from the saved mean calibration frame (aliasing calibration of main_simul,
    magnitudes/QE of ristretto_main.yml), corrected for the exposure of that frame (calibration.avgframe_steps).
    Same definition as PyrSlopec 'out_flux'."""
    frame, pup, _ = _load_calib(rMod, nSubaps, nModes, wvl, nCorrModes, seeing)
    w0, s0 = _source(rMod, nSubaps, nModes if nCorrModes is None else nCorrModes, wvl, seeing)
    steps = calibration.avgframe_steps(paths(w0).avgframe(rMod, nSubaps, s0, nModes if nCorrModes is None else nCorrModes))
    return float(np.sum(frame.ravel()[pup[pup >= 0]])) / steps

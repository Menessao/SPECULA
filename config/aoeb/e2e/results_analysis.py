import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os.path as op
import glob
import yaml
import time

import matplotlib as mpl
cmap = mpl.colormaps['hot'] # 1. Choose your colormap (e.g., 'hot', 'jet', 'turbo')
colors = cmap(np.linspace(0.0, 1.0, 7)) 
plt.rcParams['image.cmap'] = 'magma' # 4. Change the default colormap for images/scatter plots
plt.rcParams.update({
    "text.usetex": True, # Use LaTeX to write all text
    "font.family": "serif", # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Specifically use the classic LaTeX font
})
plt.rcParams['image.origin'] = 'lower'

calib_dir = '/raid1/mmenessini/calibration/WFSW'
result_dir = '/raid1/mmenessini/results/WFSW'

# Runs before CORO_NORM_FIX_TN saved the coronagraphic profile per unit of total flux, runs before CORO_AO_NORM_TN relative to the diffraction-limited peak.
# From CORO_AO_NORM_TN on it is raw contrast, relative to the peak of the integrated AO PSF, and nothing is converted. Folders without a TN (perfect_coro_*)
# are dated by the modification time of their profile file instead.
CORO_NORM_FIX_TN = '20261106_130126'
CORO_AO_NORM_TN = '20261107_120000'
pupil = fits.getdata(op.join(calib_dir, 'pupilstop', 'vlt_pupil_220pixels.fits'))
peak_DL = pupil.sum()**2 / ((pupil.shape[0] * 4)**2 * (pupil**2).sum())     # nd = 4 in all these runs

def mean_sr(result_path:str, band:str):
    """SR of a run averaged from coro_psf_<band>.start_time, the window of the integrated PSF."""
    with open(op.join(result_path, 'params.yml')) as f:
        params = yaml.safe_load(f)
    start = params[f'coro_psf_{band}'].get('start_time', 0.0) / params['main']['time_step']
    return float(np.mean(fits.getdata(op.join(result_path, f'sr_{band}.fits'))[int(round(start)):]))

def load_coro_profile(result_path:str, band:str):
    """Radial distance [lambda/D] and integrated raw contrast of a run: coronagraphic profile relative to the peak of the integrated AO PSF.
    For the older runs, which did not save that peak, the mean SR stands for it (peak of the integrated PSF / diffraction-limited peak)."""
    prof_file = op.join(result_path, f'coro_psf_{band}_int.fits')
    rad, prof = fits.getdata(prof_file)[0]
    tn = op.basename(op.normpath(result_path))
    if not tn[:8].isdigit():
        tn = time.strftime('%Y%m%d_%H%M%S', time.localtime(op.getmtime(prof_file)))
    if tn < CORO_NORM_FIX_TN:
        prof = prof / peak_DL
    if tn < CORO_AO_NORM_TN:
        prof = prof / mean_sr(result_path, band)
    return rad, prof

# Fitting-only references (perfect coronagraph, perfect 1300-mode correction), achromatic, at the conditions of the e2e sweeps (z = 30 deg, atmo.fov = 2",
# seeing 0.9" along the line of sight = 0.826" in the yml, the seeing of the s0.9 calibration products). The '_z30' folders without the seeing tag were
# run at 0.88" in the yml (0.96" along the line of sight): 2-5 lambda/D contrast 1.24x (achromatic) higher; inside the control radius the halo of an
# exact correction is second order, so it scales as seeing^(10/3), beyond it as seeing^(5/3).
# The older 'perfect_coro' (z = 0, fov = 0) carries the rotation artifact of pupil-sized layers: ~2.5x (VIS) and ~13x (NIR) worse at 2-4 lambda/D.
radp, perf_coro_vis_int = load_coro_profile(op.join(result_dir, 'perfect_coro_achrom_z30_s0.9'), 'vis')
radp, perf_coro_nir_int = load_coro_profile(op.join(result_dir, 'perfect_coro_achrom_z30_s0.9'), 'nir')

def load_perfect_coro(case:str='achrom', zenith:int=30, seeing:str='0.9'):
    """Fitting-only references at a zenith angle: case 'achrom' (no chromatism), 'vis' (WFS at 900 nm) or 'nir' (WFS at 1300 nm,
    chromatic anisoplanatism between WFS and science); seeing along the line of sight ('0.9'; None: the untagged 0.88"-in-yml runs).
    Returns rad, contrast_vis, contrast_nir."""
    path = op.join(result_dir, f'perfect_coro_{case}_z{zenith}' + (f'_s{seeing}' if seeing else ''))
    rad, cv = load_coro_profile(path, 'vis')
    rad, cn = load_coro_profile(path, 'nir')
    return rad, cv, cn

def get_latest_path(return_tn:bool=False):
    result_path = sorted(glob.glob(op.join(result_dir, '20*')))[-1]
    if return_tn:
        tn = result_path.split('/')[-1]
        return result_path, tn
    else:
        return result_path

def get_results_path(tn:str):
    return op.join(result_dir, tn)

def plot_sr(result_path:str=None):
    if result_path is None:
        result_path,tn = get_latest_path(return_tn=True)
    else:
        tn = result_path.split('/')[-1]
    srv = fits.getdata(op.join(result_path,'sr_vis.fits'))
    srn = fits.getdata(op.join(result_path,'sr_nir.fits'))
    plt.figure()
    plt.plot(srv,label='550nm')
    plt.plot(srn,label='1650nm')
    plt.ylim([0,1])
    plt.xlabel('SR')
    plt.title('frame \#')
    plt.grid()
    plt.title(tn)

def plot_coro_profiles(result_path:str=None):
    if result_path is None:
        result_path,tn = get_latest_path(return_tn=True)
    else:
        tn = result_path.split('/')[-1]
    radd, coro_psf_vis_int = load_coro_profile(result_path, 'vis')
    radd, coro_psf_nir_int = load_coro_profile(result_path, 'nir')
    plt.figure()
    plt.semilogy(radd,coro_psf_vis_int,c='C0',label='VIS')
    plt.semilogy(radd,coro_psf_nir_int,c='C3',label='NIR')
    plt.semilogy(radp,perf_coro_vis_int,'--',c='C0',label='VIS (fitting only)')
    plt.semilogy(radp,perf_coro_nir_int,'--',c='C3',label='NIR (fitting only)')
    plt.ylim([1e-7,1e-2])
    plt.xlim([0,30])
    plt.xticks(np.array([0,2,5,7.5,10,12.5,15,17.5,20,22.5,25,27.5,30]))
    plt.xlabel(r'$\lambda/D$')
    plt.ylabel('Raw contrast')
    plt.legend()
    plt.title(tn)
    plt.grid(which='both',alpha=0.4)

def modal_plot(result_path:str=None,N:int=100,title=''):
    if result_path is None:
        result_path,tn = get_latest_path(return_tn=True)
    else:
        tn = result_path.split('/')[-1]
    # dm_modes = fits.getdata(op.join(result_path,'dm_modes.fits'))
    atmo_modes = fits.getdata(op.join(result_path,'atmo_modes.fits'))
    res_modes = fits.getdata(op.join(result_path,'res_modes.fits'))
    x = np.arange(res_modes.shape[1])+1
    plt.figure()
    # plt.loglog(x,np.sqrt(np.mean(dm_modes[N:]**2,axis=0)))
    plt.loglog(x,np.sqrt(np.mean(atmo_modes[N:]**2,axis=0)),label='turbulence')
    plt.loglog(x,np.sqrt(np.mean(res_modes[N:]**2,axis=0)),label='residual')
    plt.legend()
    plt.xlabel('Mode \#')
    plt.ylabel('RMS [nm]')
    plt.grid(which='both',alpha=0.4)
    plt.title(tn)
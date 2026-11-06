import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os.path as op
import glob

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

# Coronagraphic profiles of the runs before this TN are per unit of total flux instead of contrast: they must be divided by peak_DL.
# Folders without a TN (perfect_coro_*) are treated the same way, so regenerating one with the new code would correct it twice.
CORO_NORM_FIX_TN = '20261106_130126'
pupil = fits.getdata(op.join(calib_dir, 'pupilstop', 'vlt_pupil_220pixels.fits'))
peak_DL = pupil.sum()**2 / ((pupil.shape[0] * 4)**2 * (pupil**2).sum())     # nd = 4 in all these runs

def load_coro_profile(result_path:str, band:str):
    """Radial distance [lambda/D] and integrated coronagraphic profile (peak of the diffraction-limited PSF = 1) of a run."""
    rad, prof = fits.getdata(op.join(result_path, f'coro_psf_{band}_int.fits'))[0]
    tn = op.basename(op.normpath(result_path))
    if not tn[:8].isdigit() or tn < CORO_NORM_FIX_TN:
        prof = prof / peak_DL
    return rad, prof

radp, perf_coro_vis_int = load_coro_profile(op.join(result_dir, 'perfect_coro_vis'), 'vis')
radp, perf_coro_nir_int = load_coro_profile(op.join(result_dir, 'perfect_coro_vis'), 'nir')

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
    plt.xticks(np.array([0,2,5,7,10,12,15,17,20,22,25,27,30]))
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
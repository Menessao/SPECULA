import os
import subprocess
import specula
specula.init(0)

import numpy as np
from astropy.io import fits

from specula.mmlib.yaml_overrides import write_yaml_overrides
from specula.mmlib.utils import get_pupil_mask, read_freq, r0_from_seeing
from specula.mmlib.compute_rec import compute_and_save_rec

# from specula.mmlib.save_correction_vector import save_correction_vector


rMods = np.array([0,3,4,5])
n_subaps = np.array([10,20,40])
n_modes = np.array([54,220,400])
seeings = np.array([1.0,1.5,2.0,2.5])

max_pup_dist = 60
min_pup_dist = 14

npix = 120

main_config = 'ekarus_main.yml'
root_dir='/raid1/mmenessini/calibration/ekarus'

# The yml files and temp_overrides.yml are relative paths: run from this directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_specula(calib_config):
    """Run a specula simulation, raising CalledProcessError if it fails."""
    subprocess.run(['specula', main_config, calib_config, 'temp_overrides.yml'], check=True)

# 1. Calibrate pupdata vs n_subaps
for n_subap in n_subaps:
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    if not os.path.exists(os.path.join(root_dir,'pupils',f'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits')):
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    "}")
        write_yaml_overrides(input_string=overrides)
        run_specula('calib_pupdata.yml')

# 2. Calibrate IM vs n_subaps, rMods
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        im_tag = tag+'_im'
        if not os.path.exists(os.path.join(root_dir,'im',im_tag+'.fits')):
            overrides = ("{"
                        f"pyr.pup_diam: {n_subap:.1f}, "
                        f"pyr.pup_dist: {pup_dist:.1f}, "
                        f"pyr.mod_amp: {rMod:.1f}, "
                        f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                        f"pyr_im_calibrator.im_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im', "
                        "}")
            write_yaml_overrides(input_string=overrides)
            run_specula('calib_im.yml')

for i,n_subap in enumerate(n_subaps):
    for rMod in rMods:
        tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        im_tag = tag+'_im'
        for N in n_modes[:i+1]:
            rec_tag = tag+f'_{N:1.0f}modes'        
            if not os.path.exists(os.path.join(root_dir,'rec',rec_tag+'.fits')):
                compute_and_save_rec(root_dir, im_tag=im_tag, rec_tag=rec_tag, Nmodes=N, overwrite=True)


# 3 Calibrate aliasing vs n_subap, rMods, seeing
aliaspath = os.path.join(root_dir,'aliasing')
os.makedirs(aliaspath,exist_ok=True)
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[min(i,len(n_modes)-1)]
    for rMod in rMods:
        for seeing in seeings:
            tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
            rec_tag = tag+f'_{N:1.0f}modes'
            if not os.path.exists(os.path.join(aliaspath,tag+f'_s{seeing:1.1f}_aliasing.fits')):
                overrides = ("{"
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"dm.nmodes: {N:1.0f}, "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing_random.constant: {seeing:1.1f}, "
                            "}")
                write_yaml_overrides(input_string=overrides)
                run_specula('calib_aliasing.yml')
                rec = fits.getdata(os.path.join(root_dir,'rec',rec_tag+'.fits'))
                signals = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_signals.fits'))
                frame = np.mean(fits.getdata(os.path.join(root_dir,'scratch_aliasing','frames.fits')),axis=0)

                modes = (rec @ signals.T).T
                alias_rms = np.std(modes,axis=0)

                fits.writeto(os.path.join(aliaspath,tag+f'_s{seeing:1.1f}_raw_signals.fits'),signals,overwrite=True)
                fits.writeto(os.path.join(aliaspath,tag+f'_s{seeing:1.1f}_frame_null.fits'),frame,overwrite=True)
                fits.writeto(os.path.join(aliaspath,tag+f'_s{seeing:1.1f}_aliasing.fits'),alias_rms,overwrite=True)



# 4 Calibrate SIMPC vs n_subap, rMods, seeing for PERFECT correction
ogpath = os.path.join(root_dir,'optgains')
pcimpath = os.path.join(root_dir,'pcim')
os.makedirs(ogpath,exist_ok=True)
os.makedirs(pcimpath,exist_ok=True)
ncycles = 70
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    N = n_modes[min(i,len(n_modes)-1)]
    for rMod in rMods:
        im_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_im'
        im = fits.getdata(os.path.join(root_dir,'im',im_tag+'.fits'))
        im = im[:,:N]
        im_norm = np.diag(im.T @ im)
        for seeing in seeings:
            tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.1f}'
            if not os.path.exists(os.path.join(ogpath,tag+'_og_pl.fits')):
                simpc_tag = tag+'_simpc_pl'
                temp_im_tag = tag+'_pcim_temp'
                pcim = None
                for n in range(1,ncycles+1):
                    overrides = ("{"
                                f"atmo_random.seed: {n}, "
                                f"pyr.pup_diam: {n_subap:.1f}, "
                                f"pyr.pup_dist: {pup_dist:.1f}, "
                                f"pyr.mod_amp: {rMod:.1f}, "
                                f"pushpull.nmodes: {N:1.0f}, "
                                f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                                f"dm_random.nmodes: {N:1.0f}, "
                                f"dm.nmodes: {N:1.0f}, "
                                f"main.total_time: {N*0.001*2}, "
                                f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                                f"seeing_random.constant: {seeing:1.1f}, "
                                f"pyr_im_calibrator.im_tag: '{temp_im_tag}', "
                                "}")
                    write_yaml_overrides(input_string=overrides)
                    run_specula('calib_simpc.yml')
                    if pcim is None:
                        pcim = fits.getdata(os.path.join(root_dir,'im',temp_im_tag+'.fits'))
                    else:
                        pcim += fits.getdata(os.path.join(root_dir,'im',temp_im_tag+'.fits'))
                simpc = pcim/ncycles
                fits.writeto(os.path.join(pcimpath,simpc_tag+'.fits'),simpc,overwrite=True)
                og = np.diag(simpc.T @ im)/im_norm
                fits.writeto(os.path.join(ogpath,tag+'_og_pl.fits'),og,overwrite=True)
                print('Saved optical gains as: '+tag+'_og_pl')
                # cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                # fits.writeto(os.path.join(ogpath,tag+'_compl_og_pl.fits'),cog)
                # print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og_pl')

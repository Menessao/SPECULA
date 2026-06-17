import os
import specula
specula.init(0)

import numpy as np
from astropy.io import fits

from specula.mmlib.yaml_overrides import write_yaml_overrides
# from specula.mmlib.utils import get_pupil_mask
from specula.mmlib.compute_rec import compute_and_save_rec


rMods = np.array([0,1,2,3])
n_subaps = np.array([12,16,24,48]) #np.array([16,24,48])
n_modes = np.array([50,150,300,600,1200]) #np.array([150,300,1300])
seeings = np.array([0.5,0.7,0.9,1.1,1.3,1.5]) #np.array([0.7,0.9,1.1])
max_pup_dist = 60
min_pup_dist = 16

npix = 120

main_config = 'ristretto_unobs.yml' #'ristretto_main.yml'
root_dir='/raid1/mmenessini/calibration/RISTRETTOunobs'


# # 1. Calibrate pupdata vs n_subaps
# for n_subap in n_subaps:
#     pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
#     overrides = ("{"
#                 f"pyr.pup_diam: {n_subap:.1f}, "
#                 f"pyr.pup_dist: {pup_dist:.1f}, "
#                 f"pyr_pupdata.output_tag: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
#                 "}")
#     write_yaml_overrides(input_string=overrides)
#     try:
#         os.system(f"specula {main_config} calib_pupdata.yml temp_overrides.yml")
#     except FileExistsError: #OSError:
#         pass


# 2. Calibrate IM vs n_subaps, rMods
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        pyr_tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
        pyr_im_tag = pyr_tag+'_im'        
        overrides = ("{"
                    f"pyr.pup_diam: {n_subap:.1f}, "
                    f"pyr.pup_dist: {pup_dist:.1f}, "
                    f"pyr.mod_amp: {rMod:.1f}, "
                    f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                    f"pyr_im_calibrator.im_tag: '{pyr_im_tag}', "
                    "}")
        write_yaml_overrides(input_string=overrides)
        try:
            os.system(f"specula {main_config} calib_im.yml temp_overrides.yml")
        except FileExistsError: #OSError:
            pass
        if i < len(n_subaps)-1:
            mode_vec = n_modes[:i+1]
        else:
            mode_vec = n_modes.copy()
        for N in mode_vec:
            rec_tag = pyr_tag+f'_{N:1.0f}modes'
            compute_and_save_rec(root_dir, im_tag=pyr_im_tag, rec_tag=rec_tag, Nmodes=N, overwrite=True)


# 3. Calibrate aliasing vs n_subaps, n_modes, r0
snpath = os.path.join(root_dir,'slopenulls')
aliaspath = os.path.join(root_dir,'aliasing')
framespath = os.path.join(root_dir,'frames')
os.makedirs(aliaspath,exist_ok=True)
os.makedirs(framespath,exist_ok=True)
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:     
            for N in n_modes: #mode_vec:
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}'
                Nrec = N if i == len(n_subaps)-1 else np.min((N,n_modes[i]))
                rec_tag = tag+f'_{Nrec:1.0f}modes_rec'   
                rec = fits.getdata(os.path.join(root_dir,'rec',rec_tag+'.fits'))
                overrides = ("{"
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_{Nrec:1.0f}modes_rec', "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing_random.constant: {seeing:1.2f}, "
                            f"dm_perfect.nmodes: {N:1.0f}, "
                            f"modal_analysis.nmodes: {N:1.0f}, "
                            f"source_ngs.magnitude: 5.0, "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"pyr_sn.output_tag: 'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N:1.0f}modes_sn', "
                            f"data_store.store_dir:         {os.path.join(root_dir,'scratch_aliasing')}, "  
                            f"data_store.create_tn: false, "
                            f"data_store.inputs.input_list: ['pyr_frames-cred1.out_pixels', 'pyr_modes-pyr_modalrec.out_modes'], "
                            "}")
                write_yaml_overrides(input_string=overrides)
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N:1.0f}modes'
                try:
                    sn = fits.getdata(os.path.join(snpath,tag+'_sn.fits'))
                    alias_rms = fits.getdata(os.path.join(aliaspath,tag+'_alias.fits'))
                    avg_frame = fits.getdata(os.path.join(framespath,tag+'_avg_frame.fits'))
                    print('Aliasing power and avg frame files '+tag+' already exist: skipping computation')
                except FileNotFoundError:
                    os.system(f"specula {main_config} calib_pc_sn_alias.yml temp_overrides.yml")
                    alias_modes = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_modes.fits'))
                    alias_rms = np.std(alias_modes,axis=0) #np.sqrt(np.mean(alias_modes**2,axis=0)) 
                    fits.writeto(os.path.join(aliaspath,tag+'_alias.fits'),alias_rms,overwrite=True)
                    print('Saved aliasing power as: '+tag+'_alias')
                    frames = fits.getdata(os.path.join(root_dir,'scratch_aliasing','pyr_frames.fits'))
                    frames_avg = np.mean(frames,axis=0) #np.sqrt(np.sum(frames**2,axis=0)) #
                    fits.writeto(os.path.join(framespath,tag+'_avg_frame.fits'),frames_avg,overwrite=True)
                    print('Saved average frame as: '+tag+'_avg_frame')


# 4. Calibrate SIMPC vs n_subap, rMods, seeing for PERFECT correction
ncycles = 50
fs = 2000
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:
            if i < len(n_subaps)-1:
                mode_vec = n_modes[:i+1]
            else:
                mode_vec = n_modes.copy()
            for N in mode_vec:
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N:1.0f}modes'
                simpc_tag = tag+'_simpc'
                overrides = ("{"
                            f"main.total_time: {N*2*ncycles/fs}, "
                            f"atmo_random.update_interval: {N*2:1.0f}, "
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pushpull.nmodes: {N:1.0f}, "
                            f"pushpull.ncycles: {ncycles:1.0f}, "
                            f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                            f"dm_perfect.nmodes: {N:1.0f}, "
                            f"dm.nmodes: {N:1.0f}, "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing_random.constant: {seeing:1.2f}, "
                            f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                            # f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_simpc')}', "  
                            # f"data_store.create_tn: false, "
                            # f"data_store.inputs.input_list: ['{N:1.0f}modes_pushpull-pushpull.output'], " 
                            "}")
                write_yaml_overrides(input_string=overrides)
                try:
                    os.system(f"specula {main_config} calib_perf_simpc.yml temp_overrides.yml")
                    # simpc = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
                    # og = np.diag(simpc.T @ im)/im_norm
                    # cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                    # fits.writeto(os.path.join(ogpath,tag+'_og_pl.fits'),og)
                    # print('Saved optical gains as: '+tag+'_og_pl')
                    # fits.writeto(os.path.join(ogpath,tag+'_compl_og_pl.fits'),cog)
                    # print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og_pl')
                except FileExistsError: #OSError:
                    pass

# 5. Calibrate SIMPC vs n_subap, rMods, seeing for different correction levels
ncycles = 50
fs = 2000
for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:
            if i < len(n_subaps)-1:
                mode_vec = n_modes[:i+1]
            else:
                mode_vec = n_modes.copy()
            for N in mode_vec:
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N:1.0f}modes'
                simpc_tag = tag+'_1Nm_simpc'
                overrides = ("{"
                            f"main.total_time: {N*2*ncycles/fs}, "
                            f"atmo_random.update_interval: {N*2:1.0f}, "
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pushpull.nmodes: {N:1.0f}, "
                            f"pushpull.ncycles: {ncycles:1.0f}, "
                            f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                            f"modal_analysis_random.nmodes: {N:1.0f}, "
                            f"scale_random.constant_mul_data: 'cvec_s{seeing:1.2f}_{N}modes_1Nm', "
                            f"dm_perfect.nmodes: {N:1.0f}, "
                            f"dm.nmodes: {N:1.0f}, "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing_random.constant: {seeing:1.2f}, "
                            f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                            # f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_simpc')}', "  
                            # f"data_store.create_tn: false, "
                            # f"data_store.inputs.input_list: ['{N:1.0f}modes_pushpull-pushpull.output'], " 
                            "}")
                write_yaml_overrides(input_string=overrides)
                try:
                    os.system(f"specula {main_config} calib_simpc.yml temp_overrides.yml")
                    # simpc = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
                    # og = np.diag(simpc.T @ im)/im_norm
                    # cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                    # fits.writeto(os.path.join(ogpath,tag+'_og_pl.fits'),og)
                    # print('Saved optical gains as: '+tag+'_og_pl')
                    # fits.writeto(os.path.join(ogpath,tag+'_compl_og_pl.fits'),cog)
                    # print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og_pl')
                except FileExistsError: #OSError:
                    pass

for i,n_subap in enumerate(n_subaps):
    pup_dist = np.max((min_pup_dist,max_pup_dist/max(n_subaps)*n_subap))
    for rMod in rMods:
        for seeing in seeings:
            if i < len(n_subaps)-1:
                mode_vec = n_modes[i+1]
            else:
                mode_vec = n_modes.copy()
            for N in mode_vec:
                tag = f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N:1.0f}modes'
                simpc_tag = tag+'_10Nm_simpc'
                overrides = ("{"
                            f"main.total_time: {N*2*ncycles/fs}, "
                            f"atmo_random.update_interval: {N*2:1.0f}, "
                            f"pyr.pup_diam: {n_subap:.1f}, "
                            f"pyr.pup_dist: {pup_dist:.1f}, "
                            f"pyr.mod_amp: {rMod:.1f}, "
                            f"pushpull.nmodes: {N:1.0f}, "
                            f"pushpull.ncycles: {ncycles:1.0f}, "
                            f"pyr_im_calibrator.nmodes: {N:1.0f}, "
                            f"modal_analysis_random.nmodes: {N:1.0f}, "
                            f"scale_random.constant_mul_data: 'cvec_s{seeing:1.2f}_{N}modes_10Nm', "
                            f"dm_perfect.nmodes: {N:1.0f}, "
                            f"dm.nmodes: {N:1.0f}, "
                            f"pyr_slopes.pupdata_object: 'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}', "
                            f"seeing_random.constant: {seeing:1.2f}, "
                            f"pyr_im_calibrator.im_tag: '{simpc_tag}', "
                            # f"data_store.store_dir:         '{os.path.join(root_dir,'scratch_simpc')}', "  
                            # f"data_store.create_tn: false, "
                            # f"data_store.inputs.input_list: ['{N:1.0f}modes_pushpull-pushpull.output'], " 
                            "}")
                write_yaml_overrides(input_string=overrides)
                try:
                    os.system(f"specula {main_config} calib_simpc.yml temp_overrides.yml")
                    # simpc = fits.getdata(os.path.join(root_dir,'im',simpc_tag+'.fits'))
                    # og = np.diag(simpc.T @ im)/im_norm
                    # cog = np.sqrt(np.diag(simpc.T @ simpc)/im_norm - og**2)
                    # fits.writeto(os.path.join(ogpath,tag+'_og_pl.fits'),og)
                    # print('Saved optical gains as: '+tag+'_og_pl')
                    # fits.writeto(os.path.join(ogpath,tag+'_compl_og_pl.fits'),cog)
                    # print('Saved complementary (perpedicular) optical gains as: '+tag+'_compl_og_pl')
                except FileExistsError: #OSError:
                    pass

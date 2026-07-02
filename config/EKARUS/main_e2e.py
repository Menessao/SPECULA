import os
import specula
specula.init(0)

import numpy as np
from astropy.io import fits
import glob
import pandas as pd

from specula.mmlib.yaml_overrides import write_yaml_overrides


nSubaps = np.array([10,20,30,40])
max_pup_dist = 60
min_pup_dist = 14

seeings = np.array([1.5,1.75,2.0])
freqs = np.array([1000]) #np.array([200,250,500,1000]) #250,
starMags = np.array([1,3,5,7,9,11,13])
gvec = np.arange(1,11)*0.1
# gvec = np.arange(2,14)*0.1

rMods = np.array([6,7,8]) #np.array([3,4,5,6,7,8])

init = 400
nSubap = 40
nModes = 400
pup_dist = np.max((min_pup_dist,max_pup_dist/max(nSubaps)*nSubap))
delay = 1.0e-3
savetn = False
filtertype = 'INT'

# r_vals = np.array([0.0,-0.05,-0.1,-0.15,-0.2])

main_config = 'config/EKARUS/ekarus_main.yml'
root_dir='/raid1/mmenessini/calibration/EKARUS'
result_dir='/raid1/mmenessini/results/EKARUS'


for rMod in rMods:
    results = []

    def optimize_gain(rMod,freq,starMag,seeing):
        dir = f'sao_pyr{rMod:1.0f}_{freq:1.0f}Hz_delay{delay*1e+3:1.1f}ms_s{seeing:1.1f}_mag{starMag:1.0f}'
        dirname = '/raid1/mmenessini/results/EKARUS/gain_opt/' + dir
        best_gain = 0.0
        best_sr = 0.0
        for gain in gvec:
            if filtertype == 'INT':
                store_dir = f'{dirname}/gain_{gain:.1f}'
            else:
                store_dir = f'{dirname}/iir_gain_{gain:.1f}'
            try:
                srvec = fits.getdata(os.path.join(store_dir,'sr.fits'))
            except FileNotFoundError:
                print(f'Testing gain {gain} for mag={starMag:1.1f}, {seeing:1.1f}", f={freq:1.0f}Hz, rMod={rMod:1.0f}')
                if filtertype == 'INT':
                    overrides = ("{"
                                f"main.total_time: 1.4, "
                                f"pyr.pup_diam: {nSubap:.1f}, "
                                f"pyr.pup_dist: {pup_dist:.1f}, "
                                f"seeing.constant: {seeing:1.2f}, "
                                f"pyr.mod_amp: {rMod:1.1f}, "
                                f"pyr_slopes.sn_object: pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_sn, "
                                f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_{nModes:1.0f}modes_rec', "
                                f"source_ngs.magnitude: {starMag:1.1f},"
                                f"ocam.dt: {1/freq:1.5f}, "
                                f"filter.int_gain: [{gain:.1f}], "
                                f"filter.delay: {delay*freq:1.2f}, "
                                f"data_store.store_dir: {store_dir}, "
                                f"data_store.create_tn: false, "
                                f"data_store.inputs.input_list: ['sr-psf.out_sr'], "
                                "}")
                else:
                    overrides = ("{"
                                f"main.total_time: 1.4, "
                                f"pyr.pup_diam: {nSubap:.1f}, "
                                f"pyr.pup_dist: {pup_dist:.1f}, "
                                f"seeing.constant: {seeing:1.2f}, "
                                f"pyr.mod_amp: {rMod:1.1f}, "
                                f"pyr_slopes.sn_object: pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_sn, "
                                f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_{nModes:1.0f}modes_rec', "
                                f"source_ngs.magnitude: {starMag:1.1f},"
                                f"ocam.dt: {1/freq:1.5f}, "
                                f"gain_ramp.scheduled_values: [[0.1],[{gain:1.2f}]], "
                                # f"filter.iir_filter_data_object: {filtertype}, "
                                f"filter.delay: {delay*freq:1.2f}, "
                                f"data_store.store_dir: {store_dir}, "
                                f"data_store.create_tn: false, "
                                f"data_store.inputs.input_list: ['sr-psf.out_sr'], "
                                "}")
                write_yaml_overrides(input_string=overrides)
                os.system(f"specula {main_config} temp_overrides.yml")
            srvec = fits.getdata(os.path.join(store_dir,'sr.fits'))
            sr = np.mean(srvec[init:])
            sr_std = np.std(srvec[init:])
            print(f'gain={gain:1.1f}: SR={sr:1.4f}')
            if sr > best_sr:
                best_sr = sr
                best_gain = gain
                best_sr_std = sr_std
        print(f'SR={best_sr:1.4f} for gain={best_gain:1.1f}, mag={starMag:1.1f}, {seeing:1.1f}", f={freq:1.0f}Hz, rMod={rMod:1.0f}')
        return best_gain, best_sr, best_sr_std

    for seeing in seeings:
        for freq in freqs:
            for starMag in starMags:
                # Step 1: optimize gain
                gain_opt,sr_opt,sr_std_opt = optimize_gain(rMod=rMod,freq=freq,seeing=seeing,starMag=starMag)

                # Step 2: run simulation
                if savetn:     
                    if filtertype == 'INT':               
                        overrides = ("{"
                                f"pyr.pup_diam: {nSubap:.1f}, "
                                f"pyr.pup_dist: {pup_dist:.1f}, "
                                f"seeing.constant: {seeing:1.2f}, "
                                f"pyr.mod_amp: {rMod:1.1f}, "
                                f"pyr_slopes.sn_object: pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_sn, "
                                f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_{nModes:1.0f}modes_rec', "
                                f"source_ngs.magnitude: {starMag:1.1f},"
                                f"ocam.dt: {1/freq:1.5f}, "
                                f"filter.int_gain: [{gain_opt:.1f}], "
                                f"filter.delay: {delay*freq:1.2f}, "
                                "}")
                    else:
                        overrides = ("{"
                                f"pyr.pup_diam: {nSubap:.1f}, "
                                f"pyr.pup_dist: {pup_dist:.1f}, "
                                f"seeing.constant: {seeing:1.2f}, "
                                f"pyr.mod_amp: {rMod:1.1f}, "
                                f"pyr_slopes.sn_object: pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_sn, "
                                f"pyr_modalrec.recmat_object: 'pyr{rMod:1.1f}_{nSubap:1.0f}x{nSubap:1.0f}_{nModes:1.0f}modes_rec', "
                                f"source_ngs.magnitude: {starMag:1.1f},"
                                f"ocam.dt: {1/freq:1.5f}, "
                                f"gain_ramp.scheduled_values: [[0.1],[{gain_opt:1.2f}]], "
                                # f"filter.iir_filter_data_object: {filtertype}, "
                                f"filter.delay: {delay*freq:1.2f}, "
                                "}")
                    write_yaml_overrides(input_string=overrides)
                    os.system(f"specula {main_config} temp_overrides.yml")
                    dirs = [d for d in glob.glob(os.path.join(result_dir,"20*")) if os.path.isdir(d)]
                    last_dir = sorted(dirs)[-1]
                    srvec = fits.getdata(os.path.join(last_dir,'sr.fits'))
                    tn = last_dir.split('/')[-1]
                    sr = np.mean(srvec[init:])
                    sr_std = np.std(srvec[init:])

                    # Check that sr and sr_opt match
                    if sr < sr_opt:
                        raise ValueError(f'Computed SR {sr:1.4f} is lower than the expected {sr_opt:1.4f}')

                    # Save tn and sr in a pandas dataframe
                    results.append({'seeing': seeing,'freq': freq,'starMag': starMag, 'tn': tn, 'sr': sr, 'sr_std': sr_std, 'filter': filtertype, 'gain': gain_opt})
                    columns = ['seeing', 'freq', 'starMag', 'sr', 'tn', 'gain', 'filter']
                else:
                    results.append({'seeing': seeing,'freq': freq,'starMag': starMag, 'sr': sr_opt, 'sr_std': sr_std_opt,  'filter': filtertype, 'gain': gain_opt})
                    columns = ['seeing', 'freq', 'starMag', 'sr', 'gain', 'filter']

    results_df = pd.DataFrame(results, columns=columns) 
    results_df.to_csv(os.path.join(result_dir, 'e2e_csv', f'rMod{rMod:1.0f}_IIR_{nSubap}x{nSubap}_{nModes}modes.csv'), index=False)


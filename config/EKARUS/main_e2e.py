import os
import specula
specula.init(0)

import numpy as np
from astropy.io import fits
import glob
import pandas as pd

from specula.mmlib.yaml_overrides import write_yaml_overrides

# rMods = np.array([2,3,4,5,6])
# n_subaps = np.array([10,20,30,40])
# n_modes = np.array([54,120,400])
# max_pup_dist = 60
# min_pup_dist = 14
# npix = 120

seeings = np.array([1.5,2.0,2.5])
freqs = np.array([250,500,1000])
starMags = np.array([1,3,5,7,9,11])
gvec = np.arange(1,11)*0.1

init = 400
rMod = 3

main_config = 'config/EKARUS/ekarus_main.yml'
root_dir='/raid1/mmenessini/calibration/EKARUS'
result_dir='/raid1/mmenessini/results/EKARUS'

nModes = 400

results = []

def optimize_gain(rMod,freq,starMag,seeing):
    dir = f'sao_pyr{rMod:1.0f}_{freq:1.0f}Hz_s{seeing:1.1f}_mag{starMag:1.0f}'
    dirname = '/raid1/mmenessini/results/EKARUS/gain_opt/' + dir
    best_gain = 0.0
    best_sr = 0.0
    for gain in gvec:
        store_dir = f'{dirname}/gain_{gain:.1f}'
        try:
            srvec = fits.getdata(os.path.join(store_dir,'sr.fits'))
        except FileNotFoundError:
            print(f'Testing gain {gain} for mag={starMag:1.1f}, {seeing:1.1f}", f={freq:1.0f}Hz, rMod={rMod:1.0f}')
            overrides = ("{"
                        f"filter.int_gain: [{gain:.1f}], "
                        f"data_store.store_dir: {store_dir}, "
                        f"data_store.create_tn: false"
                        "}")
            write_yaml_overrides(input_string=overrides)
            os.system(f"specula {main_config} temp_overrides.yml")
        srvec = fits.getdata(os.path.join(store_dir,'sr.fits'))
        sr = np.mean(srvec[init:])
        print(f'gain={gain:1.1f}: SR={sr:1.4f}')
        if sr > best_sr:
            best_sr = sr
            best_gain = gain
    print(f'SR={best_sr:1.4f} for gain={best_gain:1.1f}, mag={starMag:1.1f}, {seeing:1.1f}", f={freq:1.0f}Hz, rMod={rMod:1.0f}')
    return best_gain, best_sr

for seeing in seeings:
    for freq in freqs:
        for starMag in starMags:
            # Step 1: optimize gain
            gain_opt,sr_opt = optimize_gain(rMod=rMod,freq=freq,seeing=seeing,starMag=starMag)

            # Step 2: run simulation
            overrides = ("{"
                    f"filter.int_gain: [{gain_opt:.1f}], "
                    "}")
            write_yaml_overrides(input_string=overrides)
            os.system(f"specula {main_config} temp_overrides.yml")
            dirs = [d for d in glob.glob(os.path.join(result_dir,"20*")) if os.path.isdir(d)]
            last_dir = sorted(dirs)[-1]
            srvec = fits.getdata(os.path.join(last_dir,'sr.fits'))
            tn = last_dir.split('/')[-1]
            sr = np.mean(srvec[init:])

            # Check that sr and sr_opt match
            if sr < sr_opt:
                raise ValueError(f'Computed SR {sr:1.4f} is lower than the expected {sr_opt:1.4f}')

            # Save tn and sr in a pandas dataframe
            results.append({
                'seeing': seeing,
                'freq': freq,
                'starMag': starMag,
                'tn': tn,
                'sr': float(sr)
            })

results_df = pd.DataFrame(results, columns=['seeing', 'freq', 'starMag', 'tn', 'sr'])
results_df.to_csv(os.path.join(result_dir, f'e2e_rMod{rMod:1.0f}_{nModes}modes.csv'), index=False)










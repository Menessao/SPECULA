import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides
from astropy.io import fits
import pandas as pd

# Range of gains to test
gains1 = np.linspace(0.1, 1.0, 10) 
gains2 = np.linspace(0.0, 1.0, 11) 
base_config = "config/SensorFusion/ristretto_faint_CAO.yml"

freqs = np.array([1000,2000,4000])
delay2 = 250e-6
init = 1600

overrides_name = 'temp_CAO_overrides'

results = []
result_dir = '/raid1/mmenessini/results/Cascading'
# sr_mat = np.zeros([len(gains),len(gains)])

for freq in freqs:
    dirname = f'{result_dir}/cao_gain_opt/cao_f2_{freq/1000:1.0f}kHz'
    for gain1 in gains1:
        for gain2 in gains2:
            store_dir = f'{dirname}/int1gain{gain1:.2f}_int2gain{gain2:.2f}'
            try:
                srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
            except FileNotFoundError:
                print(f'Testing gains: {gain1:1.2f},{gain2:1.2f}')
                overrides = ("{"
                            f"main.total_time: 1.0, "
                            f"cred2.dt: {1/freq:1.6f}, "
                            f"filter1.int_gain: [{gain1:.2f}], "
                            f"filter2.int_gain: [{gain2:.2f}], "
                            f"filter2.delay: {delay2*freq:1.2f}, "
                            f"data_store.store_dir: {store_dir}, "
                            f"data_store.create_tn: false, "
                            f"data_store.inputs.input_list: ['sr1-coro_psf1.out_sr', 'sr2-coro_psf2.out_sr'], "
                            "}")
                write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
                os.system(f"specula {base_config} {overrides_name}.yml")
                srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
            sr = np.mean(srvec[init:])                 
            print(f'SR={sr:1.4f} for CAO: INT1 gain={gain1:1.1f}, INT2 gain={gain2:1.1f}, f2={freq/1000:1.0f}kHz')  
            results.append({'freq': freq, 'gain1': gain1, 'gain2': gain2, 'sr': sr})
            columns = ['freq', 'sr', 'gain1', 'gain2']

results_df = pd.DataFrame(results, columns=columns) 
results_df.to_csv(os.path.join(result_dir, 'e2e_csv', 'ristretto_faint_CAO.csv'), index=False)
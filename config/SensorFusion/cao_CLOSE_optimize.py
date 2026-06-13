import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides
from astropy.io import fits
import pandas as pd

base_config = "config/SensorFusion/ristretto_faint_CAO.yml"

overrides_name = 'temp_CAO_overrides'

results = []
result_dir = '/raid1/mmenessini/results/Cascading'
# sr_mat = np.zeros([len(gains),len(gains)])


# Range of gains to test
gains1 = np.array([0.01,0.1,0.2,0.3,0.4,0.5])
freqs1 = np.array([100,500,1000])
freqs2 = np.array([1000,2000,4000])
r_vals = np.array([0.0,-0.05,-0.1,-0.15,-0.2,-0.25,-0.3])
delay = 250e-6
init = 6000

for freq1 in freqs1:
    dirname = f'{result_dir}/cao_gain_opt_CLOSE'
    for gain1 in gains1:
        for freq2 in freqs2:
            for r_val in r_vals:
                store_dir = f'{dirname}/int1gain{gain1:.2f}_{freq1:1.0f}Hzf1_{freq2:1.0f}Hzf2_r{r_val:1.2f}'
                try:
                    srvec = fits.getdata(os.path.join(store_dir,f'sr2.fits'))
                except FileNotFoundError:
                    print(f'Testing gain: g1={gain1:1.2f},f1={freq1:1.0f}Hz,f2={freq2:1.0f}Hz')
                    overrides = ("{"
                                f"main.total_time: 3.0, "
                                f"cred1.dt: {1/freq1:1.6f}, "
                                f"cred2.dt: {1/freq2:1.6f}, "
                                f"filter1.int_gain: [{gain1:.2f},{gain1/2:.2f}], "
                                f"filter2.int_gain: [1.0], "
                                f"filter1.delay: {delay*freq1:1.2f}, "
                                f"filter2.delay: {delay*freq2:1.2f}, "
                                f"omgi.dt: {(2*(delay*freq2+1)-1):1.2f},"
                                f"omgi.r: {r_val:1.2f}, "
                                f"data_store.store_dir: '{store_dir}', "
                                f"data_store.create_tn: false, "
                                f"data_store.inputs.input_list: ['sr1-coro_psf1.out_sr', 'sr2-coro_psf2.out_sr'], "
                                "}")
                    write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
                    os.system(f"specula {base_config} {overrides_name}.yml")
                    srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
                sr = np.mean(srvec[init:])                 
                print(f'SR={sr:1.4f} for CAO: INT1 gain={gain1:1.1f}, f1={freq1:1.0f}Hz, f2={freq2/1000:1.0f}kHz')  
                results.append({'freq1': freq1, 'freq2': freq2, 'gain1': gain1, 'gain2': 'CLOSE', 'sr': sr, 'r': r_val})
                columns = ['freq1','freq2', 'sr', 'gain1', 'gain2','r']

results_df = pd.DataFrame(results, columns=columns) 
results_df.to_csv(os.path.join(result_dir, 'e2e_csv', 'ristretto_faint_CAO_CLOSE.csv'), index=False)

# # Range of gains to test
# gains1 = np.linspace(0.1, 1.0, 10) 
# gains2 = np.linspace(0.0, 1.0, 11) 
# freqs = np.array([1000,2000,4000])
# delay2 = 250e-6
# init = 1600

# for freq in freqs:
#     dirname = f'{result_dir}/cao_gain_opt/cao_f2_{freq/1000:1.0f}kHz'
#     for gain1 in gains1:
#         for gain2 in gains2:
#             store_dir = f'{dirname}/int1gain{gain1:.2f}_int2gain{gain2:.2f}'
#             try:
#                 srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
#             except FileNotFoundError:
#                 print(f'Testing gains: {gain1:1.2f},{gain2:1.2f}')
#                 overrides = ("{"
#                             f"main.total_time: 1.0, "
#                             f"cred2.dt: {1/freq:1.6f}, "
#                             f"filter1.int_gain: [{gain1:.2f}], "
#                             f"filter2.int_gain: [{gain2:.2f}], "
#                             f"filter2.delay: {delay2*freq:1.2f}, "
#                             f"data_store.store_dir: {store_dir}, "
#                             f"data_store.create_tn: false, "
#                             f"data_store.inputs.input_list: ['sr1-coro_psf1.out_sr', 'sr2-coro_psf2.out_sr'], "
#                             "}")
#                 write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
#                 os.system(f"specula {base_config} {overrides_name}.yml")
#                 srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
#             sr = np.mean(srvec[init:])                 
#             print(f'SR={sr:1.4f} for CAO: INT1 gain={gain1:1.1f}, INT2 gain={gain2:1.1f}, f2={freq/1000:1.0f}kHz')  
#             results.append({'freq': freq, 'gain1': gain1, 'gain2': gain2, 'sr': sr})
#             columns = ['freq', 'sr', 'gain1', 'gain2']

# results_df = pd.DataFrame(results, columns=columns) 
# results_df.to_csv(os.path.join(result_dir, 'e2e_csv', 'ristretto_faint_CAO.csv'), index=False)

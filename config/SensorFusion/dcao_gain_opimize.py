import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides
from astropy.io import fits
import pandas as pd

# Range of gains to test
gains = np.linspace(0.1, 1.0, 10) 
base_config = "config/SensorFusion/ristretto_faint_dCAO.yml"

freqs = np.array([1000,2000,4000])
delay2 = 250e-6
init = 1600

overrides_name = 'temp_dCAO_overrides'

results = []
result_dir = '/raid1/mmenessini/results/Cascading'
f1name = 'iirfilter_1300modes_exc1_pow1.5'
f2name = 'iirfilter_150modes_exc2_pow1.0'

dirname = f'{result_dir}/dcao_gain_opt/dcao_first_stage'
sr_best = 0.0
best_gain1 = 0.0
for gain1 in gains:
    store_dir = f'{dirname}/int1gain{gain1:.2f}_{f1name}'
    try:
        srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
    except FileNotFoundError:
        print(f'Testing IIR1 gain: {gain1:1.2f}')
        overrides = ("{"
                    f"main.total_time: 1.0, "
                    f"bootstrap1.scheduled_values: [[0.05],[{gain1:1.2f}]], "
                    f"bootstrap2.scheduled_values: [[0.0],[0.0]], "
                    f"filter1.iir_filter_data_object: {f1name}, "
                    f"data_store.store_dir: {store_dir}, "  
                    f"dm2.inputs.in_command:      'filter2.out_comm', " # disable dCAO
                    f"data_store.create_tn: false, "
                    f"data_store.inputs.input_list: ['sr1-coro_psf1.out_sr', 'sr2-coro_psf2.out_sr'], "
                    "}")
        write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
        os.system(f"specula {base_config} {overrides_name}.yml")
        srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
    sr = np.mean(srvec[init:])                 
    print(f'SR={sr:1.4f} for dCAO: INT1 gain={gain1:1.1f}')
    results.append({'freq2': None, 'gain1': gain1, 'gain2': 0.0, 'sr': sr, 'IIR1': f1name, 'IIR2': f2name })
    columns = ['freq', 'sr', 'gain1', 'gain2', 'IIR1', 'IIR2']
    if sr > sr_best:
        sr_best = sr
        best_gain1 = gain1
print(f'Selected IIR1 gain = {best_gain1:1.2f}, yielding SR = {sr_best:1.4f}')

for freq in freqs:
    dirname = f'{result_dir}/dcao_gain_opt/dcao_f2_{freq/1000:1.0f}kHz'
    for gain2 in gains:
        store_dir = f'{dirname}/int2gain{gain2:.2f}_{f1name}_{f2name}'
        try:
            srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
        except FileNotFoundError:
            print(f'Testing IIR2 gain: {gain2:1.2f} (IIR1 gain = {best_gain1:1.2f})')
            overrides = ("{"
                        f"main.total_time: 1.0, "
                        f"cred2.dt: {1/freq:1.6f}, "
                        f"bootstrap1.scheduled_values: [[0.05],[{best_gain1:1.2f}]], "
                        f"bootstrap2.scheduled_values: [[0.1],[{gain2:1.2f}]], "
                        f"filter1.iir_filter_data_object: {f1name}, "
                        f"filter2.iir_filter_data_object: {f2name}, "
                        f"filter2.delay: {delay2*freq:1.2f}, "
                        f"dm2.inputs.in_command: 'dcao_cmd2.out_value', " # enable dCAO
                        f"data_store.store_dir: {store_dir}, "
                        f"data_store.create_tn: false, "
                        f"data_store.inputs.input_list: ['sr1-coro_psf1.out_sr', 'sr2-coro_psf2.out_sr'], "
                        "}")
            write_yaml_overrides(input_string=overrides, temp_name=overrides_name)
            os.system(f"specula {base_config} {overrides_name}.yml")
            srvec = fits.getdata(os.path.join(store_dir,'sr2.fits'))
        sr = np.mean(srvec[init:])                 
        print(f'SR={sr:1.4f} for dCAO: IIR1 gain={best_gain1:1.1f}, IIR2 gain={gain2:1.1f}, f2={freq/1000:1.0f}kHz')  
        results.append({'freq': freq, 'gain1': best_gain1, 'gain2': gain2, 'sr': sr, 'IIR1': f1name, 'IIR2': f2name })
        columns = ['freq', 'sr', 'gain1', 'gain2', 'IIR1', 'IIR2']

results_df = pd.DataFrame(results, columns=columns) 
results_df.to_csv(os.path.join(result_dir, 'e2e_csv', 'ristretto_faint_dCAO.csv'), index=False)
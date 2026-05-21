# import specula

import os
import numpy as np
from specula.mmlib.utils import radial_order
from specula.data_objects.iir_filter_data import IirFilterData

import matplotlib.pyplot as plt
import scipy.signal as signal

def save_and_test_restore(filter_data_complex:IirFilterData, file_name:str):
    filter_data_complex.save(file_name)
    print(f"Saved with native method: {file_name}")
    try:
        loaded_filter_native = IirFilterData.restore(file_name)
        print(f"Loaded: {loaded_filter_native.nfilter} filters")
        coeffs_match = np.allclose(loaded_filter_native.num, filter_data_complex.num) and \
                      np.allclose(loaded_filter_native.den, filter_data_complex.den)
        print(f"Matching filters: {coeffs_match}")
    except FileNotFoundError:
        print("File FITS not found")


def create_stepped_t(n_filters:int, excluded_filters:int=None):
    n = radial_order(n_filters)
    if excluded_filters is not None:
        n_min = radial_order(excluded_filters)
    else:
        n_min = 0
    n_max = radial_order(n_filters)
    rad_t = np.zeros(int(n))
    rad_t[n_min:] = np.linspace(1/n_max, 1, int(n - n_min))
    t = np.hstack([np.repeat(rad_t[i-2],i) for i in range(2, len(rad_t)+2)])
    t = t[:n_filters]  # Ensure we only have n_filters elements
    return t


def guidos_standard_iir(n_filters:int, excluded_filters:int,
                        start_pole = [1.0, 0.995],
                        end_pole = [0.9, 0.75],
                        start_zero = [0.85, 0.45],
                        end_zero = [0.55, 0.30],
                        power_exponent = 2.0, iir_gain=1.0):
    t = create_stepped_t(n_filters,excluded_filters=excluded_filters)
    t_powered = t**power_exponent 

    end_pole[0] = start_pole[0] - 0.1 * n_filters/1000
    end_pole[1] = start_pole[1] - 0.245 * n_filters/1000

    end_zero[0] = start_zero[0] - 0.3 * n_filters/1000
    end_zero[1] = start_zero[1] - 0.15 * n_filters/1000

    zero_values = start_zero[0] + (end_zero[0] - start_zero[0]) * t_powered
    zero2_values = start_zero[1] + (end_zero[1] - start_zero[1]) * t_powered
    pole_values = start_pole[0] + (end_pole[0] - start_pole[0]) * t_powered
    pole2_values = start_pole[1] + (end_pole[1] - start_pole[1]) * t_powered

    num_list = []
    den_list = []
    for i in range(n_filters):
        num_list.append([zero_values[i]*zero2_values[i],
                         -1*(zero_values[i]+zero2_values[i]), 
                         1.0])
        den_list.append([pole_values[i]*pole2_values[i], 
                         -pole_values[i]-pole2_values[i], 1.0])

    num_array = np.array(num_list)
    den_array = np.array(den_list)

    num_array *= iir_gain
    return num_array, den_array


def plot_iir_tfs(filter_data_complex:IirFilterData, fs:float, n_filters:int, delay_frames:float=2.0):
    nw_delay, dw_delay = filter_data_complex.discrete_delay_tf(delay_frames)
    freq = np.logspace(-2, np.log10(fs/2), 2000)    
    plt.figure(figsize=(16,5))
    modes = np.round(10**np.linspace(0,np.log10(n_filters),5)-1).astype(int)
    for mode in modes:
        rtf = filter_data_complex.RTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        ntf = filter_data_complex.NTF(mode=mode, fs=fs, freq=freq, dm=1.0, nw=nw_delay, dw=dw_delay, plot=False)
        plt.subplot(1,2,1)
        plt.loglog(freq,rtf,label=f'Mode {mode}')
        plt.subplot(1,2,2)
        plt.loglog(freq,ntf,label=f'Mode {mode}')
    plt.subplot(1,2,1)
    plt.legend()
    plt.grid(which='both',alpha=0.3)
    plt.xlim([1e-1,fs/2])
    plt.xlabel('Frequency [Hz]')
    plt.title('RTF')
    plt.subplot(1,2,2)
    plt.legend()
    plt.grid(which='both',alpha=0.3)
    plt.xlim([1e-1,fs/2])
    plt.xlabel('Frequency [Hz]')
    plt.title('NTF')

    plt.show()


if __name__ == "__main__":

    root_dir = '/raid1/mmenessini/calibration/XAO'
    # root_dir = '/raid1/mmenessini/calibration/SOUL/KLv30dx'
    # root_dir = '/raid1/mmenessini/calibration/EKARUS'
    path = os.path.join(root_dir,'filter')
    os.makedirs(path,exist_ok=True)

    fs = 1000  # sampling frequency
    n_filters = 1300
    excluded_filters = 0
    make_tiled = False
    file_name = os.path.join(path,f'cascading_iirfilter_{n_filters}modes.fits')

    num_array,den_array=guidos_standard_iir(n_filters=n_filters,
                                            excluded_filters=excluded_filters,
                                            power_exponent=2.0) # used 0.8 for EKARUS
    
    # b,a=design_f3_controller(fs=2000,f1=10,f2=300,N=2)

    filter_data_complex = IirFilterData(
        ordnum=[3] * n_filters,
        ordden=[3] * n_filters,
        num=num_array,
        den=den_array
    )
    save_and_test_restore(filter_data_complex,file_name)

    if make_tiled:
        tiled_file_name = path + f'tiled_iirfilter_{n_filters}modes.fits'
        tiled_filter_data_complex = IirFilterData(
            ordnum=[3] * n_filters*2,
            ordden=[3] * n_filters*2,
            num=np.tile(num_array,[2,1]),
            den=np.tile(den_array,[2,1])
        )
        save_and_test_restore(tiled_filter_data_complex,tiled_file_name)

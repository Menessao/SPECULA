import specula
specula.init(0)

import numpy as np
from astropy.io import fits
import os
from specula.data_objects.spatio_temp_array import SpatioTempArray


def save_spatiotemp_array(dir_path:str, phs_file:str, tag:str, dt:float):
    phs = fits.getdata(phs_file)
    print(phs.shape)
    tvec = np.arange(phs.shape[0])*dt
    sta = SpatioTempArray(phs,time_vector=tvec,time_axis=0)
    
    cube_dir = os.path.join(dir_path,'arrays')
    os.makedirs(cube_dir,exist_ok=True)
    fname = os.path.join(cube_dir,tag+'.fits')
    sta.save(fname)
    print(f'✅ Saved: {fname}')
    return fname
    
if __name__ == "__main__":
    dir_path = '/raid1/mmenessini/calibration/SOUL/KLv30dx'
    tag = '20230507_030749_res_OPD_fit'
    phs_file = '/raid1/mmenessini/calibration/SOUL/KLv30dx/data/20230507_030749_res_OPD_fit.fits'
    dt = 0.000588
    save_spatiotemp_array(dir_path=dir_path, phs_file=phs_file, tag=tag, dt=dt)
    tag = '20230507_030749_res_OPD'
    phs_file = '/raid1/mmenessini/calibration/SOUL/KLv30dx/data/20230507_030749_res_OPD.fits'
    save_spatiotemp_array(dir_path=dir_path, phs_file=phs_file, tag=tag, dt=dt)
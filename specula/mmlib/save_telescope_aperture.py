import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

from astropy.io import fits
from specula.lib.toccd import toccd
import os
import sys

import matplotlib.pyplot as plt

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from specula.lib.make_mask import make_mask

def save_copernico_pupil(destination_dir:str, tag:str, Npix=160, obs=0.3, angle=-30, D:float=1.82, overwrite=False):
    new_pupil = make_mask(np_size=Npix,obsratio=obs,spider=True,n_petals=4,angle_offset=angle,spider_width=0.02/D*Npix)
    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, tag+f'_{Npix:1.0f}pixels.fits')
    fits.writeto(fname,new_pupil.reshape([Npix,Npix,1]),overwrite=overwrite)
    return new_pupil

def save_pupil_to_size(data_dir:str, destination_dir:str, tag:str, Npix:int, thr:float=0.9, D:float=8.2):
    hdu = fits.open(os.path.join(data_dir, tag+f'_{Npix}pixels.fits'))
    data = hdu[0].data
    pupil = data[:,:,0]
    new_pupil = toccd(pupil,(Npix,Npix),xp=specula.xp)
    new_pupil = new_pupil >= thr*new_pupil.max()
    # new_pupil = specula.xp.array(new_pupil,dtype=specula.float_dtype)
    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, tag+f'_{Npix:1.0f}pixels.fits')
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=new_pupil)
    pupilstop.save(fname)
    return new_pupil

def save_lbt_pupil(destination_dir:str='/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils', Npix:int=120, D:float=8.222):
    pupil = fits.getdata('/raid1/mmenessini/LBTData/lbtpupilcrop.fits')
    os.makedirs(destination_dir,exist_ok=True)
    new_pupil = toccd(pupil,(Npix,Npix),xp=specula.xp)
    new_pupil = new_pupil >= 0.5*new_pupil.max()
    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, f'lbt_pupil_{Npix:1.0f}pixels.fits')
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=new_pupil)
    pupilstop.save(fname)
    return new_pupil

if __name__ == "__main__":

    if len(sys.argv) > 1:
        Npix = int(sys.argv[1])
    else:
        Npix = 160
    if len(sys.argv) > 2:
        obs = float(sys.argv[2])
    else:
        obs = 0.3
    if len(sys.argv) > 3:
        angle = float(sys.argv[3])
    else:
        angle = -30

    # save_lbt_pupil(Npix=220)

    destination_dir = '/raid1/mmenessini/calibration/EKARUS/pupilstop'
    tag = 'Copernico_Pupil'
    save_copernico_pupil(destination_dir=destination_dir, tag=tag, obs=obs, angle=angle, Npix=Npix, overwrite=True)
    aperture=save_pupil_to_size(destination_dir, destination_dir, tag, Npix, D=1.82)

    # data_dir = '/raid1/mmenessini/calibration/VLT'
    # destination_dir = '/raid1/mmenessini/calibration/XAO/pupilstop'
    # tag = 'vlt_pupil'
    # Npix = 160
    # aperture=save_pupil_to_size(data_dir, destination_dir, tag, Npix, thr=0.69)
    # plt.figure()
    # plt.imshow(aperture,origin='lower',cap='gray')
    # plt.show()

    # destination_dir = '/raid1/mmenessini/calibration/EKARUS/pupilstop'
    # tag = 'copernico_pupil'
    # Npix = 160
    # save_copernico_pupil(destination_dir, tag, overwrite=True)
    # aperture=save_pupil_to_size(destination_dir, destination_dir, tag, Npix, D=1.82)
    # plt.figure()
    # plt.imshow(aperture,origin='lower',cmap='gray')
    # plt.show()




from astropy.io import fits
import numpy as np
import os.path as op

def generate_rec(im,Nmodes:int,argos:bool,iir_path:str='/raid1/mmenessini/calibration/SOUL/KLv30dx/data/iir_rows.fits'):
    IM = np.zeros_like(im)
    # Adjust slopes
    IM[:2512//2,:] = im[2512//2:,:]
    IM[2512//2:,:] = im[:2512//2,:]*-1
    # Adjust scaling
    IM *= 4e+9
    iir_rows = fits.getdata(iir_path)
    pad_iir_rows = np.pad(iir_rows,pad_width=((600-Nmodes,0),(0,0)),mode='constant',constant_values=0.0)
    IMinv = np.linalg.pinv(IM[:2512,:Nmodes])
    IMinv = np.pad(IMinv,pad_width=((0,0),(0,2848-2512)),mode='constant',constant_values=0.0)
    Rec = np.vstack([IMinv,pad_iir_rows])
    Rec = Rec.astype('>f4')
    if argos:
        Rec /= 2
    return Rec


if __name__ == "__main__":
    impath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/im'
    recpath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/rec'
    tag = 'pyr3.0_40x40_shift'
    Nmodes = np.array([200,300,400,500])
    print(tag)
    IM = fits.getdata(op.join(impath,tag+'_im.fits'))
    for N in Nmodes:
        Rec = generate_rec(IM,N,argos=True)
        fits.writeto(op.join(recpath,tag+'_rec.fits'),Rec,overwrite=True)
        print(f'Saved rec as {tag}_{N}modes_rec.fits')

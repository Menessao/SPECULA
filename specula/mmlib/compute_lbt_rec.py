
from astropy.io import fits
import numpy as np
import os.path as op


pyr_masks = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupmask_shift.fits').astype(bool)
pupids = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/pup_ids.fits')
filepath=f'/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupdata.fits'
rec_hdr = fits.getheader('/raid1/mmenessini/calibration/SOUL/KLv30dx/data/Rec_LUCI2_IIR_bin1_500modes.fits')
npix = 120
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data
fimg = np.zeros(npix**2)

def generate_rec(im,Nmodes:int,argos:bool): #,iir_path:str='/raid1/mmenessini/calibration/SOUL/KLv30dx/data/iir_rows.fits'
    aux = np.zeros_like(im)
    IM = np.zeros_like(im)
    # Adjust slopes
    aux[:2512//2,:] = im[2512//2:,:]
    aux[2512//2:,:] = im[:2512//2,:]*-1
    # Adjust scaling
    aux *= 4e+9
    half_mask = pyr_masks[:60,:120].astype(bool)
    img = np.zeros(np.size(half_mask))
    for i in range(aux.shape[1]):
        np.put(fimg, pup_ids[:,0], aux[:2512//2,i])
        np.put(fimg, pup_ids[:,1], aux[2512//2:,i])
        f2d = fimg.reshape([npix,npix])
        img = f2d[:60,:]
        IM[pupids,i] = img.flatten()[half_mask.flatten()]
    IMinv = np.linalg.pinv(IM[:2512,:Nmodes])
    Rec = np.pad(IMinv,pad_width=((672-Nmodes,0),(0,2848-2512)),mode='constant',constant_values=0.0)
    # Add IIR rows
    Rec[661,:] = IMinv[0,:].copy()    
    Rec[668,:] = IMinv[1,:].copy()
    Rec = Rec.astype('>f4')
    if argos:
        Rec /= 2
    return Rec,IM


if __name__ == "__main__":
    impath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/im'
    recpath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/rec'
    tag = 'pyr3.0_s1.0_synim' #'pyr3.0_40x40_shift_im' #'
    Nmodes = np.array([100,200,300,400,500,600])
    print(tag)
    IntMat = fits.getdata(op.join(impath,tag+'.fits'))
    for N in Nmodes:
        Rec,IM = generate_rec(IntMat,N,argos=True)
        fits.writeto(op.join(recpath,tag+f'_lbtlike_im.fits'),IM,header=rec_hdr,overwrite=True)
        print(f'Saved im as {tag}_lbtlike_im.fits')
        fits.writeto(op.join(recpath,tag+f'_{N}modes_rec.fits'),Rec,header=rec_hdr,overwrite=True)
        print(f'Saved rec as {tag}_{N}modes_rec.fits')

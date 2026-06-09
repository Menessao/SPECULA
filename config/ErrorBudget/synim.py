from astropy.io import fits
import numpy as np
from scipy.ndimage import rotate
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv

klinv = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_kl_inv.fits')
kl = np.linalg.pinv(klinv)
ifunc = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc.fits')
lbtpup = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupilstop/asm_v30dx_197pixels.fits')
im = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbt_refim.fits')
pupids = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/pup_ids.fits')
pyr_masks = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupmask_shift.fits').astype(bool)

filepath=f'/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupdata.fits'

def rotate_ifunc(rot_deg:float):
    ifunc_new = np.zeros_like(ifunc)
    kl_new = np.zeros_like(kl)
    img = np.zeros(lbtpup.shape)
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        rot_img = rotate(img,angle=rot_deg,reshape=False)
        ifunc_new[:,j] = rot_img[lbtpup.astype(bool)]
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=lbtpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_shift.fits', overwrite=True)
    for j in range(kl.shape[1]):
        img[lbtpup.astype(bool)] = kl[:,j]
        rot_img = rotate(img,angle=rot,reshape=False)
        kl_new[:,j] = rot_img[lbtpup.astype(bool)]
    kl_inv_new = np.linalg.pinv(kl_new)
    ifunc_inv_obj = IFuncInv(ifunc_inv=kl_inv_new, mask=lbtpup)
    ifunc_inv_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_shift_inv.fits', overwrite=True)

Nslopes = 2512
Nmodes = 400
npix = 120

half_mask = pyr_masks[:60,:120].astype(bool)
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data
fimg = np.zeros(npix**2)


def evaluate_error():
    refim = im[:Nslopes,:Nmodes]
    aux = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbt_synim.fits')
    synim = aux.copy()
    synim[:Nslopes//2,:] = aux[Nslopes//2:,:]
    synim[Nslopes//2:,:] = aux[:Nslopes//2,:]*-1

    synim -= np.mean(synim,axis=0)
    synim *= np.std(refim,axis=0)/np.std(synim,axis=0)

    err = np.zeros(Nmodes)
    for j in Nmodes:
        img = np.zeros(np.size(half_mask))
        img[half_mask.flatten()] = refim[pupids,j]
        img = img.reshape([60,120])
        np.put(fimg, pup_ids[:,0], synim[:Nslopes//2,j])
        np.put(fimg, pup_ids[:,1], synim[Nslopes//2:,j])
        f2d = fimg.reshape([npix,npix])
        delta = img - f2d[:60,:120]
        err[j] = np.sum(delta[half_mask]**2)
    return err
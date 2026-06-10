from astropy.io import fits
import numpy as np
import os
import pandas as pd

import specula 
specula.init(0)

from scipy.ndimage import rotate
from specula.data_objects.ifunc import IFunc

# from specula.data_objects.ifunc_inv import IFuncInv
# from scipy.ndimage import zoom

klinv = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_kl_inv.fits')
kl = np.linalg.pinv(klinv)
ifunc = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc.fits')
lbtpup = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupilstop/asm_v30dx_197pixels.fits')
im = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbt_refim.fits')
pupids = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/pup_ids.fits')
pyr_masks = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupmask_shift.fits').astype(bool)

filepath=f'/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupdata.fits'


# def change_magnification(image, factor):
#     scaled = zoom(image, factor, order=3)
#     old_h, old_w = image.shape
#     new_h, new_w = scaled.shape
#     if factor >= 1.0:
#         start_y = (new_h - old_h) // 2
#         start_x = (new_w - old_w) // 2
#         return scaled[start_y:start_y + old_h, start_x:start_x + old_w]
#     else:
#         output = np.zeros_like(image)
#         start_y = (old_h - new_h) // 2
#         start_x = (old_w - new_w) // 2
#         output[start_y:start_y + new_h, start_x:start_x + new_w] = scaled
#         return output

def shift_image(image, shift, axis):
    shift_int = int(np.floor(shift))
    shift_frac = shift - shift_int
    def integer_shift(img, pixels, ax):
        if pixels == 0:
            return img.copy()
        pad_width = [(0, 0), (0, 0)]
        if pixels > 0:
            pad_width[ax] = (pixels, 0)  # Pad start (left/top)
            sliced_img = np.pad(img, pad_width, mode='constant', constant_values=0)
            if ax == 0: return sliced_img[:-pixels, :]
            else:       return sliced_img[:, :-pixels]
        else:
            pad_width[ax] = (0, abs(pixels))  # Pad end (right/bottom)
            sliced_img = np.pad(img, pad_width, mode='constant', constant_values=0)
            if ax == 0: return sliced_img[abs(pixels):, :]
            else:       return sliced_img[:, abs(pixels):]
    img_floor = integer_shift(image, shift_int, axis)
    img_ceil = integer_shift(image, shift_int + 1, axis)
    shifted_image = img_floor * (1.0 - shift_frac) + img_ceil * shift_frac
    return shifted_image

# def rotate_ifunc_inv(rot_deg:float):
#     kl_new = np.zeros_like(kl)
#     img = np.zeros(lbtpup.shape)
#     for j in range(kl.shape[1]):
#         img[lbtpup.astype(bool)] = kl[:,j]
#         rot_img = rotate(img,angle=rot_deg,reshape=False)
#         kl_new[:,j] = rot_img[lbtpup.astype(bool)]
#     kl_inv_new = np.linalg.pinv(kl_new)
#     ifunc_inv_obj = IFuncInv(ifunc_inv=kl_inv_new, mask=lbtpup)
#     ifunc_inv_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_optshift_inv.fits', overwrite=True)

def rotate_ifunc(rot_deg:float):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(lbtpup.shape)
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        rot_img = rotate(img,angle=rot_deg,reshape=False)
        ifunc_new[:,j] = rot_img[lbtpup.astype(bool)]
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=lbtpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_optshift.fits', overwrite=True)

def shift_ifunc(shift:float,ax_dir):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(lbtpup.shape)
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        shift_img = shift_image(img,shift=shift,axis=ax_dir)
        ifunc_new[:,j] = shift_img[lbtpup.astype(bool)]
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=lbtpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_optshift.fits', overwrite=True)

Nslopes = 2512
npix = 120

half_mask = pyr_masks[:60,:120].astype(bool)
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data
fimg = np.zeros(npix**2)

def evaluate_error(Nmodes:int):
    refim = im[:Nslopes,:Nmodes]
    aux = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbt_optsynim.fits')[:,:Nmodes]
    synim = aux.copy()
    synim[:Nslopes//2,:] = aux[Nslopes//2:,:]
    synim[Nslopes//2:,:] = aux[:Nslopes//2,:]*-1
    synim -= np.mean(synim,axis=0)
    synim *= np.std(refim,axis=0)/np.std(synim,axis=0)
    err = np.zeros(Nmodes)
    for j in range(Nmodes):
        img = np.zeros(np.size(half_mask))
        img[half_mask.flatten()] = refim[pupids,j]
        img = img.reshape([60,120])
        np.put(fimg, pup_ids[:,0], synim[:Nslopes//2,j])
        np.put(fimg, pup_ids[:,1], synim[Nslopes//2:,j])
        f2d = fimg.reshape([npix,npix])
        delta = img - f2d[:60,:120]
        err[j] = np.sqrt(np.sum(delta[half_mask]**2))
    return err

def evaluate_metric(Nmodes):
    main_config = 'syn_soul_im.yml'
    os.system(f"specula {main_config}")
    err = evaluate_error(Nmodes)
    metric = np.sqrt(np.sum(err**2))
    return metric



if __name__ == "__main__":

    Nmodes = 500

    rot0 = 52.5
    shiftX0 = 0.0
    shiftY0 = 0.0

    rotvec = np.linspace(-2,2,15)
    shiftvec = np.linspace(-1,1,21)

    result_dir = '/raid1/mmenessini/results/SOUL/KLv30dx/'

    columns = ['rotation','shiftX','shiftY','magnification','metric']
    result = []

    print('Setting default parameters')
    shift_ifunc(shiftX0,ax_dir=0)
    shift_ifunc(shiftY0,ax_dir=1)

    for rot in rotvec:
        print(f'Testing rotation: {rot}')
        rotate_ifunc(rot_deg=rot+rot0)
        chi = evaluate_metric(Nmodes)
        result.append({'rotation': rot+rot0, 'shiftX': shiftX0, 'shiftY': shiftY0, 'magnification': 1.00, 'metric': chi})
        print(f'Obtained metric: {chi}')
    rotate_ifunc(rot_deg=rot0)

    for shft in shiftvec:
        print(f'Testing x-shift: {shft}')
        shift_ifunc(shft+shiftX0,ax_dir=0)
        chi = evaluate_metric(Nmodes)
        result.append({'rotation': rot0, 'shiftX': shft+shiftX0, 'shiftY': shiftY0, 'magnification': 1.00, 'metric': chi})
        print(f'Obtained metric: {chi}')
    shift_ifunc(shiftX0,ax_dir=0)

    for shft in shiftvec:
        print(f'Testing y-shift: {shft}')
        shift_ifunc(shft+shiftY0,ax_dir=1)
        chi = evaluate_metric(Nmodes)
        result.append({'rotation': rot0, 'shiftX': shiftX0, 'shiftY': shft+shiftY0, 'magnification': 1.00, 'metric': chi})
        print(f'Obtained metric: {chi}')
    shift_ifunc(shiftY0,ax_dir=1)

    results_df = pd.DataFrame(result, columns=columns) 
    results_df.to_csv(os.path.join(result_dir, 'misreg_csv', f'deltaRot{np.max(abs(rotvec)):1.2f}_deltaShift{np.max(abs(shiftvec)):1.2f}_metric.csv'), index=False)
        


                


from astropy.io import fits
import numpy as np
import os
import pandas as pd

import specula 
specula.init(0)

from skimage.transform import AffineTransform,warp
from scipy.ndimage import rotate #,zoom
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

main_config = 'syn_soul_im.yml'

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

def warp_image(ifunc,shear:float=0,rot:float=0,mag:float=1.0):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(lbtpup.shape)
    center_y, center_x = img.shape[0]/2.0, img.shape[1]/2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    shear_and_scale = AffineTransform(shear=shear, rotation=rot, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x, center_y))
    trf = shift_to_origin + shear_and_scale + shift_to_center
    # warp_mask = np.logical_or(warp((1-lbtpup), inverse_map=trf.inverse),(1-lbtpup).astype(bool))
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        warp_img = warp(img, inverse_map=trf.inverse)
        ifunc_new[:,j] = warp_img[lbtpup.astype(bool)]
    return ifunc_new

def rotate_ifunc(ifunc,rot_deg:float):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(lbtpup.shape)
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        rot_img = rotate(img,angle=rot_deg,reshape=False)
        ifunc_new[:,j] = rot_img[lbtpup.astype(bool)]
    return ifunc_new

def shift_ifunc(ifunc,shift:float,ax_dir):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(lbtpup.shape)
    for j in range(ifunc.shape[1]):
        img[lbtpup.astype(bool)] = ifunc[:,j]
        shift_img = shift_image(img,shift=shift,axis=ax_dir)
        ifunc_new[:,j] = shift_img[lbtpup.astype(bool)]
    return ifunc_new


def set_ifunc_pars(ifunc,shiftX=None,shiftY=None,rot=None,mag=None,shearAmp=None,shearAngle=0):
    ifunc_new = ifunc.copy()
    if rot is not None:
        ifunc_new[:] = rotate_ifunc(ifunc_new,rot_deg=rot)
    if shiftX is not None:
        ifunc_new[:] = shift_ifunc(ifunc_new,shift=shiftX,ax_dir=0)
    if shiftY is not None:
        ifunc_new[:] = shift_ifunc(ifunc_new,shift=shiftY,ax_dir=1)
    if mag is not None:
        ifunc_new[:] = warp_image(ifunc_new,mag=mag)
    if shearAmp is not None:
        ifunc_new[:] = warp_image(ifunc_new,shear=shearAmp,rot=shearAngle)
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=lbtpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_optshift.fits', overwrite=True)

def save_ifunc_pars(ifunc,shiftX=None,shiftY=None,rot=None,mag=None,shearAmp=None,shearAngle=0):
    ifunc_new = ifunc.copy()
    ifunc_inv_new = klinv.copy()
    if rot is not None:
        ifunc_new[:] = rotate_ifunc(ifunc_new,rot_deg=rot)
        ifunc_inv_new[:] = rotate_ifunc(ifunc_inv_new.T,rot_deg=rot).T
    if shiftX is not None:
        ifunc_new[:] = shift_ifunc(ifunc_new,shift=shiftX,ax_dir=0)
        ifunc_inv_new[:] = shift_ifunc(ifunc_inv_new.T,shift=shiftX,ax_dir=0).T
    if shiftY is not None:
        ifunc_new[:] = shift_ifunc(ifunc_new,shift=shiftY,ax_dir=1)
        ifunc_inv_new[:] = shift_ifunc(ifunc_inv_new.T,shift=shiftY,ax_dir=1).T
    if mag is not None:
        ifunc_new[:] = warp_image(ifunc_new,mag=mag)
        ifunc_inv_new[:] = warp_image(ifunc_inv_new,mag=mag).T
    if shearAmp is not None:
        ifunc_new[:] = warp_image(ifunc_new,shear=shearAmp,rot=shearAngle)
        ifunc_inv_new[:] = warp_image(ifunc_inv_new,shear=shearAmp,rot=shearAngle).T
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=lbtpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_shift.fits', overwrite=True)
    ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv_new.T,mask=lbtpup)
    ifunc_inv_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc_shift_inv.fits', overwrite=True)

Nslopes = 2512
npix = 120

half_mask = pyr_masks[:60,:120].astype(bool)
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data

def get_synim(Nmodes:int,alpha=None):
    if alpha is not None:
        set_ifunc_pars(ifunc,rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3])
        os.system(f"specula {main_config}")
    aux = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbt_synim.fits')[:,:Nmodes]
    synim = aux.copy()
    synim[:Nslopes//2,:] = aux[Nslopes//2:,:]
    synim[Nslopes//2:,:] = aux[:Nslopes//2,:]*-1
    # synim -= np.mean(synim,axis=0)
    # synim *= np.std(refim,axis=0)/np.std(synim,axis=0)
    fimg = np.zeros(npix**2)
    synim_true = np.zeros([Nslopes,Nmodes])
    for j in range(Nmodes):
        np.put(fimg, pup_ids[:,0], synim[:Nslopes//2,j])
        np.put(fimg, pup_ids[:,1], synim[Nslopes//2:,j])
        f2d = fimg.reshape([npix,npix])
        fcut = f2d[:60,:120]
        synim_true[:,j] = fcut[half_mask]
    return synim_true

def get_refim(Nmodes:int):
    refim = im[:Nslopes,:Nmodes]
    refim_true = np.zeros([Nslopes,Nmodes])
    for j in range(Nmodes):
        img = np.zeros(np.size(half_mask))
        img[half_mask.flatten()] = refim[pupids,j]
        img = img.reshape([60,120])
        refim_true[:,j] = img[half_mask]
    return refim_true



# def evaluate_error(Nmodes:int):
#     refim = im[:Nslopes,:Nmodes]
#     synim = get_synim(Nmodes)
#     err = np.zeros(Nmodes)
#     for j in range(Nmodes):
#         img = np.zeros(np.size(half_mask))
#         img[half_mask.flatten()] = refim[pupids,j]
#         img = img.reshape([60,120])
#         np.put(fimg, pup_ids[:,0], synim[:Nslopes//2,j])
#         np.put(fimg, pup_ids[:,1], synim[Nslopes//2:,j])
#         f2d = fimg.reshape([npix,npix])
#         delta = img - f2d[:60,:120]
#         err[j] = np.sqrt(np.sum(delta[half_mask]**2))
#     return err

def sensitivity_matrix(alphas,eps_vec,Nmodes):
    sens = []
    print('Computing sensitivity matrix')
    for k,eps in enumerate(eps_vec):
        alpha_eps = alpha.copy()
        alpha_eps[k] += eps
        push = get_synim(Nmodes,alpha_eps)
        alpha_eps[k] -= 2*eps
        pull = get_synim(Nmodes,alpha_eps)
        delta = (push-pull)/(2*eps)
        sens.append(delta.flatten())
    sens = np.array(sens).T
    return sens


delta_vec = lambda vec: (np.max(vec)-np.min(vec))/len(vec)


# rot0 = 55.38
# shiftX0 = -0.627
# shiftY0 = -0.848

if __name__ == "__main__":

    rot0 = 54
    shiftX0 = 0.0
    shiftY0 = 0.0
    mag0 = 1.0

    drot = 0.5
    dshft = 0.01
    dmag = 0.0025

    result_dir = '/raid1/mmenessini/results/SOUL/KLv30dx/'
    Nmodes = 500

    tol = 1e-2
    max_its = 10
    
    alpha = np.array([rot0,shiftX0,shiftY0,mag0])
    eps = np.array([drot,dshft,dshft,dmag])
    refim = get_refim(Nmodes)
    err = tol + 1
    k = 0

    while err > tol and k < max_its:
        print(f'Iteration {k}')
        synim = get_synim(Nmodes,alpha=alpha)
        sens = sensitivity_matrix(alpha,eps,Nmodes)

        # Update gain
        G = np.diag(np.linalg.pinv(synim) @ refim)

        # Update alpha
        aux = ((refim @ np.diag(1/G)) - synim)
        dalpha = np.linalg.pinv(sens) @ aux.flatten()
        print(f'Update parameters are: {dalpha}')
        alpha_new = alpha + dalpha
        err = np.max(np.abs(dalpha)/np.abs(alpha))

        # Update synim
        alpha = alpha_new
        k += 1
    
    if k == max_its:
        print(f'\nOptimization did not converge in {max_its} iterations! Last parameters: {alpha}')
    else:
        print(f'\nOptimization success in {k} iterations! Found parameters: {alpha}')
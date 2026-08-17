from astropy.io import fits
import numpy as np
import os

import specula 
specula.init(0)

from skimage.transform import AffineTransform,warp
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv

from specula.mmlib.save_telescope_aperture import save_pupil

klinv = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_kl_inv.fits')
kl = np.linalg.pinv(klinv)

ifunc = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_ifunc.fits')
lbtpup = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupilstop/asm_v30dx_197pixels.fits')

rec = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/data/Rec_20251120_085351.fits')
im = np.linalg.pinv(rec)

pupids = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/pup_ids.fits')
pyr_masks = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupmask_shift.fits').astype(bool)

filepath=f'/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupdata.fits'

main_config = 'syn_soul_im.yml'

def warp_image(ifunc,pupmask,
               flip:bool=False,
               shftX:float=0.0,shftY:float=0.0,
               shear:float=0,rot:float=0,
               mag:float=1.0,
               oldpup=lbtpup):
    pup_mask = pupmask.astype(bool)
    ifunc_new = np.zeros([int(np.sum(pup_mask)),ifunc.shape[1]])
    img = np.zeros(lbtpup.shape)
    center_y, center_x = img.shape[0]/2.0, img.shape[1]/2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    shear_and_scale = AffineTransform(shear=shear, rotation=rot*np.pi/180, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x+shftX, center_y+shftY))
    trf = shift_to_origin + shear_and_scale + shift_to_center
    for j in range(ifunc.shape[1]):
        img[oldpup.astype(bool)] = ifunc[:,j]
        if flip:
            img = img[::-1,:]
        warp_img = warp(img, inverse_map=trf.inverse)
        ifunc_new[:,j] = warp_img[pup_mask]
    return ifunc_new

def warp_mask(pup,shftX:float=0.0,shftY:float=0.0,mag:float=1.0,rot:float=0.0):
    center_y, center_x = pup.shape[0]/2.0, pup.shape[1]/2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    scale = AffineTransform(rotation=rot*np.pi/180,scale=mag)
    shift_to_center = AffineTransform(translation=(center_x+shftX, center_y+shftY))
    trf = shift_to_origin + scale + shift_to_center
    warp_pup = (warp(pup.astype(float), inverse_map=trf.inverse)) > 0.9
    return warp_pup.astype(float)


def set_ifunc_pars(flip=True,shiftX=0.0,shiftY=0.0,rot=0.0,mag=1.0,shearAmp=None,shearAngle=0):
    warpup = warp_mask(lbtpup,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    if shearAmp is not None:
        oldpup = warpup.copy()
        warpup[:] = warp_mask(warpup,shear=shearAmp,rot=shearAngle)
        ifunc_new[:] = warp_image(ifunc_new,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup)
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_lbti_ifunc_optshift.fits', overwrite=True)
    save_pupil(warpup, '/raid1/mmenessini/calibration/SOUL/KLv30dx/pupilstop/', fname='asm_v30dx_197pixels_optshift', Npix=197, D=8.222)


def save_ifunc_pars(flip=True,shiftX=0.0,shiftY=0.0,rot=0.0,mag=1.0,shearAmp=None,shearAngle=0):
    warpup = warp_mask(lbtpup,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_inv_new = warp_image(klinv.T,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag).T
    if shearAmp is not None:
        oldpup = warpup.copy()
        warpup[:] = warp_mask(warpup,shear=shearAmp,rot=shearAngle)
        ifunc_new[:] = warp_image(ifunc_new,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup)
        ifunc_inv_new[:] = warp_image(ifunc_inv_new.T,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup).T
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_lbti_ifunc_shift.fits', overwrite=True)
    ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv_new.T,mask=warpup)
    ifunc_inv_obj.save('/raid1/mmenessini/calibration/SOUL/KLv30dx/ifunc/asm_v30dx_lbti_ifunc_shift_inv.fits', overwrite=True)
    save_pupil(warpup, '/raid1/mmenessini/calibration/SOUL/KLv30dx/pupilstop/', fname='asm_v30dx_197pixels_shift', Npix=197, D=8.222)


Nslopes = 2512
npix = 120

half_mask = pyr_masks[:60,:120].astype(bool)
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data

def get_synim(Nmodes:int,alpha=None):
    if alpha is not None:
        set_ifunc_pars(rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3])
        os.system(f"specula {main_config} synim_overrides.yml")
    aux = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/im/pyr3.0_40x40_lbti_synim.fits')[:,:Nmodes]
    synim = aux.copy()
    synim[:Nslopes//2,:] = aux[Nslopes//2:,:]
    synim[Nslopes//2:,:] = aux[:Nslopes//2,:]*-1
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


def sensitivity_matrix(alphas,eps_vec,Nmodes):
    sens = []
    print('Computing sensitivity matrix')
    for k,eps in enumerate(eps_vec):
        alpha_eps = alphas.copy()
        alpha_eps[k] += eps
        push = get_synim(Nmodes,alpha_eps)
        alpha_eps[k] -= 2*eps
        pull = get_synim(Nmodes,alpha_eps)
        delta = (push-pull)/(2*eps)
        sens.append(delta.flatten())
    sens = np.array(sens).T
    return sens

    # rot0 = -34.08
    # shiftX0 = -0.59
    # shiftY0 = 0.12
    # mag0 = 0.981

if __name__ == "__main__":

    rot0 = -34.1
    shiftX0 = 0.27
    shiftY0 = -1.01
    mag0 = 0.974

    drot = 0.2
    dshft = 0.01
    dmag = 0.001

    result_dir = '/raid1/mmenessini/results/SOUL/KLv30dx/'
    Nmodes = 500

    tol = 0
    max_its = 10

    alpha = np.array([rot0,shiftX0,shiftY0,mag0])
    # synim = get_synim(Nmodes=Nmodes,alpha=alpha)
    # print('Done')
    eps = np.array([drot,dshft,dshft,dmag])
    refim = get_refim(Nmodes)
    err = tol + 1
    k = 0

    while err > tol and k < max_its:
        print(f'Iteration {k}')
        sens = sensitivity_matrix(alpha,eps,Nmodes)

        # Update gain and synIM
        synim = get_synim(Nmodes,alpha=alpha)
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
        save_ifunc_pars(rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3])
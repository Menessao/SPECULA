from astropy.io import fits
import numpy as np
import os
import pandas as pd

import specula 
specula.init(0)

from skimage.transform import AffineTransform,warp

from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.mmlib.utils import get_pupil_mask, get_frame_pupil_centers, shift_image
from specula.mmlib.yaml_overrides import write_yaml_overrides

from specula.mmlib.save_telescope_aperture import save_pupil

klinv = fits.getdata('/raid1/mmenessini/calibration/EKARUS/ifunc/reordered_unobs_DM468_kl_inv.fits')
kl = np.linalg.pinv(klinv)
ifunc = fits.getdata('/raid1/mmenessini/calibration/EKARUS/ifunc/reordered_unobs_DM468_ifunc.fits')
ekapup = fits.getdata('/raid1/mmenessini/calibration/EKARUS/pupilstop/reordered_unobs_DM468_160pixels.fits')

# imfull = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260802_233704.fits')
imfull = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260731_233429.fits')
pyr_mask = get_pupil_mask(npix=240,filepath='/raid1/mmenessini/calibration/EKARUS/pupils/pyr_pupdata_onbench.fits')
crop_pyr_mask = pyr_mask[60:180,60:180]

pup_hdu = fits.open('/raid1/mmenessini/calibration/EKARUS/pupils/pyr_pupdata_onbench.fits')
pup_ids = pup_hdu[1].data

rMod = 5.0
im_tag = f'pyr{rMod:1.1f}_dm468_onbench_synim'
ifunc_tag = 'dm468_ifunc_shift'
m2c_tag = 'M2C_KL_OOPAO_central_obstruction'
m2c_tag = 'M2C_KL_OOPAO_synthetic'


def warp_image(ifunc,pupmask,
               flip:bool=False,
               shftX:float=0.0,shftY:float=0.0,
               shear:float=0,rot:float=0,
               mag:float=1.0,
               oldpup=ekapup):
    pup_mask = pupmask.astype(bool)
    ifunc_new = np.zeros([int(np.sum(pup_mask)),ifunc.shape[1]])
    img = np.zeros(ekapup.shape)
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
    warpup = warp_mask(ekapup,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    if shearAmp is not None:
        oldpup = warpup.copy()
        warpup[:] = warp_mask(warpup,shear=shearAmp,rot=shearAngle)
        ifunc_new[:] = warp_image(ifunc_new,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup)
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save(f'/raid1/mmenessini/calibration/EKARUS/ifunc/{ifunc_tag}.fits', overwrite=True)
    save_pupil(warpup, '/raid1/mmenessini/calibration/EKARUS/pupilstop/', fname='DM468_160pixels_shift', Npix=160, D=1.82)


def save_ifunc_pars(flip=True,shiftX=0.0,shiftY=0.0,rot=0.0,mag=1.0,shearAmp=None,shearAngle=0):
    warpup = warp_mask(ekapup,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag)
    ifunc_inv_new = warp_image(klinv.T,warpup,flip=flip,shftX=shiftX,shftY=shiftY,rot=rot,mag=mag).T
    if shearAmp is not None:
        oldpup = warpup.copy()
        warpup[:] = warp_mask(warpup,shear=shearAmp,rot=shearAngle)
        ifunc_new[:] = warp_image(ifunc_new,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup)
        ifunc_inv_new[:] = warp_image(ifunc_inv_new.T,warpup,shear=shearAmp,rot=shearAngle,oldpup=oldpup).T
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/EKARUS/ifunc/dm468_ifunc_bestshift.fits', overwrite=True)
    ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv_new.T,mask=warpup)
    ifunc_inv_obj.save('/raid1/mmenessini/calibration/EKARUS/ifunc/dm468_ifunc_bestshift_inv.fits', overwrite=True)
    save_pupil(warpup, '/raid1/mmenessini/calibration/EKARUS/pupilstop/', fname='DM468_160pixels_bestshift', Npix=160, D=1.82)

imframe = np.std(imfull,axis=1).reshape([240,240])
ref_centers = get_frame_pupil_centers(imframe)
avg_center = np.mean(ref_centers,axis=0)

hsize = 120
refim = np.zeros([np.sum(crop_pyr_mask),imfull.shape[1]])
for j in range(imfull.shape[1]):
    img = imfull[:,j].reshape([240,240])
    auximg = shift_image(img, shift=120-avg_center[1], axis=0)
    frimg = shift_image(auximg, shift=120-avg_center[0], axis=1)
    crop_img = frimg[60:180,60:180]
    refim[:,j] = crop_img[crop_pyr_mask]

def get_synim(Nmodes:int,alpha=None):
    if alpha is not None:
        set_ifunc_pars(rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3])
        main_config = 'ekarus_onbench.yml calib_im.yml'
        os.system(f"specula {main_config} temp_synim.yml")
    calibim = fits.getdata(f'/raid1/mmenessini/calibration/EKARUS/im/{im_tag}.fits')[:,:Nmodes]
    calibim -= np.mean(calibim,axis=0)
    calibim *= np.std(refim,axis=0)/np.std(calibim,axis=0)
    synim = np.zeros(refim.shape)
    for j in range(Nmodes):
        fimg = np.zeros([240,240])
        rtot = 0
        for i in range(4):
            valid_ids = pup_ids[:,i] != -1
            len_ids = np.sum(valid_ids)
            np.put(fimg, pup_ids[valid_ids,i], calibim[rtot:rtot+len_ids,j])
            rtot += len_ids
        f2d = fimg.reshape([240,240])[60:180,60:180]
        synim[:,j] = f2d[crop_pyr_mask]
    return synim

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


if __name__ == "__main__":

    Nmodes = 400
    ovdes = ("{"
            f"main.total_time: {Nmodes*0.001*2}, "
            f"dm.nmodes: {Nmodes}, "
            f"pushpull.nmodes: {Nmodes}, "  
            f"pushpull.amp:    25, "
            f"pupilstop.tag: 'DM468_160pixels_shift', "
            f"pyr_im_calibrator.nmodes: {Nmodes}, "
            f"pyr_im_calibrator.im_tag: {im_tag}, "
            f"pyr_im_calibrator.overwrite: true, "
            f"pyr.mod_amp: {rMod:1.1f}, "
            f"dm.ifunc_object:      {ifunc_tag}, "
            f"dm.m2c_object:        {m2c_tag}, "
            "}")
    write_yaml_overrides(input_string=ovdes, temp_name='temp_synim')

    rot0 = 0.0
    shiftX0 = 0.0
    shiftY0 = 0.0
    mag0 = 1.0

    drot = 0.2
    dshft = 0.01
    dmag = 0.001

    tol = 0
    max_its = 10

    alpha = np.array([rot0,shiftX0,shiftY0,mag0])
    # synim = get_synim(Nmodes=Nmodes,alpha=alpha)
    # print('Done')
    eps = np.array([drot,dshft,dshft,dmag])
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
        # err = np.max(np.abs(dalpha)-np.abs(eps)/2)

        # Update synim
        alpha = alpha_new
        k += 1
    
    if k == max_its:
        print(f'\nOptimization did not converge in {max_its} iterations! Last parameters: {alpha}')
    else:
        print(f'\nOptimization success in {k} iterations! Found parameters: {alpha}')
        save_ifunc_pars(rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3])

                


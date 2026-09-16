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

klinv = fits.getdata('/raid1/mmenessini/calibration/EKARUS/ifunc/meas_unobs_DM468_kl_inv.fits')
kl = np.linalg.pinv(klinv)
ifunc = fits.getdata('/raid1/mmenessini/calibration/EKARUS/ifunc/meas_unobs_DM468_ifunc.fits').T

# imfull = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260802_233704.fits')
# imfull = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260731_233429.fits')
imfull = fits.getdata('/raid1/mmenessini/calibration/EKARUS/data/IntMat_20260904_103513.fits')
pyr_mask = get_pupil_mask(npix=240,filepath='/raid1/mmenessini/calibration/EKARUS/pupils/pyr_pupdata_onbench.fits')
crop_pyr_mask = pyr_mask[60:180,60:180]
pup_hdu = fits.open('/raid1/mmenessini/calibration/EKARUS/pupils/pyr_pupdata_onbench.fits')
pup_ids = pup_hdu[1].data

og_ekapup = (fits.getdata('/raid1/mmenessini/calibration/EKARUS/pupilstop/meas_unobs_DM468_160pixels.fits')).astype(bool)
ekapup = og_ekapup.copy()
# ekapup = np.logical_and(warp_mask(og_ekapup,shftX=0,shftY=0,mag=0.97).astype(bool),og_ekapup.astype(bool)).astype(float)
# ifunc = remap_on_new_mask(ifunc,(1-og_ekapup).astype(bool),(1-ekapup).astype(bool))

rMod = 5.0
im_tag = f'pyr{rMod:1.1f}_dm468_onbench_synim'
ifunc_tag = 'dm468_ifunc_shift'
m2c_tag = 'M2C_KL_OOPAO_central_obstruction'
m2c_tag = 'M2C_KL_OOPAO_synthetic'

def warp_mask(pup,shftX:float=0.0,shftY:float=0.0,mag:float=1.0,rot:float=0.0,scaleX:float=1.0,scaleY:float=1.0,phi:float=0):
    center_y, center_x = pup.shape[0]/2.0, pup.shape[1]/2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    shift_to_coords = AffineTransform(translation=(center_x + shftX, center_y + shftY))
    scale_at_angle = (
        AffineTransform(rotation=-phi*np.pi/180)
        + AffineTransform(scale=(scaleX * mag, scaleY * mag))
        + AffineTransform(rotation=phi*np.pi/180)
    )
    rotation = AffineTransform(rotation=rot*np.pi/180)
    trf = shift_to_origin + scale_at_angle + rotation + shift_to_coords
    warp_pup = (warp(pup.astype(float), inverse_map=trf.inverse)) > 0.1
    return warp_pup.astype(float)

def warp_image(ifunc,pupmask,
               flip:bool=False,
               shftX:float=0.0,shftY:float=0.0,
               scaleX:float=1.0,scaleY:float=1.0,
               phi:float=0,
               rot:float=0,
               mag:float=1.0,
               oldpup=ekapup):
    pup_mask = pupmask.astype(bool)
    ifunc_new = np.zeros([int(np.sum(pup_mask)),ifunc.shape[1]])
    img = np.zeros(ekapup.shape)
    center_y, center_x = img.shape[0]/2.0, img.shape[1]/2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    shift_to_coords = AffineTransform(translation=(center_x + shftX, center_y + shftY))
    scale_at_angle = (
        AffineTransform(rotation=-phi*np.pi/180)
        + AffineTransform(scale=(scaleX * mag, scaleY * mag))
        + AffineTransform(rotation=phi*np.pi/180)
    )
    rotation = AffineTransform(rotation=rot*np.pi/180)
    trf = shift_to_origin + scale_at_angle + rotation + shift_to_coords
    for j in range(ifunc.shape[1]):
        img[oldpup.astype(bool)] = ifunc[:,j]
        if flip:
            img = img[::-1,:]
        warp_img = warp(img, inverse_map=trf.inverse)
        ifunc_new[:,j] = warp_img[pup_mask.astype(bool)]
    return ifunc_new

def set_ifunc_pars(flip=False,shiftX=0.0,shiftY=0.0,rot=0.0,mag=1.0,scaleX=1.0,scaleY=1.0,phi=0.0):
    auxpup = np.logical_and(og_ekapup,warp_mask(ekapup,shftX=shiftX,shftY=shiftY,mag=mag))
    warpup = warp_mask(auxpup,rot=rot,scaleX=scaleX,scaleY=scaleY,phi=phi)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,rot=rot,scaleX=scaleX,scaleY=scaleY,phi=phi)
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save(f'/raid1/mmenessini/calibration/EKARUS/ifunc/{ifunc_tag}.fits', overwrite=True)
    save_pupil(warpup, '/raid1/mmenessini/calibration/EKARUS/pupilstop/', fname='DM468_160pixels_shift', Npix=160, D=1.82)


def save_ifunc_pars(flip=False,shiftX=0.0,shiftY=0.0,rot=0.0,mag=1.0,scaleX=1.0,scaleY=1.0,phi=0.0):
    auxpup = np.logical_and(og_ekapup,warp_mask(ekapup,shftX=shiftX,shftY=shiftY,mag=mag))
    warpup = warp_mask(auxpup,rot=rot,scaleX=scaleX,scaleY=scaleY,phi=phi)
    ifunc_new = warp_image(ifunc,warpup,flip=flip,rot=rot,scaleX=scaleX,scaleY=scaleY,phi=phi)
    ifunc_inv_new = warp_image(klinv,warpup,flip=flip,rot=rot,scaleX=scaleX,scaleY=scaleY,phi=phi).T
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=warpup)
    ifunc_obj.save('/raid1/mmenessini/calibration/EKARUS/ifunc/dm468_ifunc_bestshift.fits', overwrite=True)
    ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv_new,mask=warpup)
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

def get_synim(Nmodes:int,flip=False,alpha=None):
    if alpha is not None or flip:
        set_ifunc_pars(flip=flip,rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],
                       mag=alpha[3],scaleX=alpha[4],scaleY=alpha[5],phi=alpha[0])
        main_config = 'ekarus_onbench.yml calib_im.yml'
        os.system(f"specula {main_config} temp_synim.yml")
    calibim = fits.getdata(f'/raid1/mmenessini/calibration/EKARUS/im/{im_tag}.fits')[:,:Nmodes]
    synim = np.zeros([refim.shape[0],Nmodes])
    for j in range(Nmodes):
        fimg = np.zeros([240,240])
        pup_ids_full = np.stack([pup_ids[:,0],pup_ids[:,1],pup_ids[:,2],pup_ids[:,3]])
        pup_ids_full = pup_ids_full[pup_ids_full>-1]
        np.put(fimg, pup_ids_full, calibim[:,j])
        f2d = fimg.reshape([240,240])[60:180,60:180]
        synim[:,j] = f2d[crop_pyr_mask]
    return synim

def sensitivity_matrix(alphas,eps_vec,Nmodes,flip):
    sens = []
    print('Computing sensitivity matrix')
    for k,eps in enumerate(eps_vec):
        alpha_eps = alphas.copy()
        alpha_eps[k] += eps
        push = get_synim(Nmodes,alpha=alpha_eps,flip=flip)
        alpha_eps[k] -= 2*eps
        pull = get_synim(Nmodes,alpha=alpha_eps,flip=flip)
        delta = (push-pull)/(2*eps)
        sens.append(delta.flatten())
    sens = np.array(sens).T
    return sens

def write_overrides(Nmodes,tlt_coeffs):
    ovdes = ("{"
            f"main.total_time: {Nmodes*0.001*2}, "
            f"dm.nmodes: {Nmodes}, "
            f"pushpull.nmodes: {Nmodes}, "  
            f"pushpull.amp:    200, "
            f"pupilstop.tag: 'DM468_160pixels_shift', "
            f"pyr_im_calibrator.nmodes: {Nmodes}, "
            f"pyr_im_calibrator.im_tag: {im_tag}, "
            f"pyr_im_calibrator.overwrite: true, "
            f"pyr.mod_amp: {rMod:1.1f}, "
            f"pyr.pyr_tlt_coeff: {tlt_coeffs.tolist()}, "
            f"dm.ifunc_object:      {ifunc_tag}, "
            f"dm.m2c_object:        {m2c_tag}, "
            "}")
    write_yaml_overrides(input_string=ovdes, temp_name='temp_synim')


def pupil_sensitivity_matrix(tlts,dtlt,Nmodes,alpha,flip):
    sens = []
    print('Computing pupil sensitivity matrix')
    tlt_vec = tlts.flatten()
    print(tlt_vec)
    for k,_ in enumerate(tlt_vec):
        dtlt_vec = tlt_vec.copy()
        dtlt_vec[k] += dtlt
        write_overrides(Nmodes,dtlt_vec.reshape([2,4]))
        push = get_synim(Nmodes,alpha=alpha,flip=flip)
        dtlt_vec[k] -= 2*dtlt
        write_overrides(Nmodes,dtlt_vec.reshape([2,4]))
        pull = get_synim(Nmodes,alpha=alpha,flip=flip)
        delta = (push-pull)/(2*dtlt)
        sens.append(delta.flatten())
    sens = np.array(sens).T
    return sens


if __name__ == "__main__":

    # tlt_coeffs = np.array([[1.137576415499041, 1.107481227742432, 1.1601173385737144, 1.1221432137981107],
    # [1.0463793131565175, 0.9524373022367241, 1.0519332127629741, 0.9738255619117916]])
    tlt_coeffs = np.array([[1.1227635531778277, 1.0711816924375064, 1.1748214310491363, 1.1250855697126243],
    [0.9997977924653967, 0.8922379224358873, 1.0854080667722075, 1.037476552728351]])

    flip = True
    rot0 = -89
    shiftX0 = 0.0
    shiftY0 = 0.0
    mag0 = 0.97
    scaleX0 = 1.0
    scaleY0 = 0.989

    drot = 0.25
    dshft = 0.01
    dmag = 0.001
    dshear = 0.001

    tol = 1e-2
    max_its = 30

    doShear = False

    alpha = np.array([rot0,shiftX0,shiftY0,mag0,scaleX0,scaleY0])
    eps = np.array([drot,dshft,dshft,dmag,dshear])
    if doShear is False:
        eps = eps[:4]

    dtlt = 0.001
    opt_tlt_coeffs = tlt_coeffs.copy()

    Nmodes = 200
    err = tol + 1
    k = 0
    while err > tol and k < max_its:
        print(f'Iteration {k}')

        # Step 1: optimize pupil position
        # if k > 0:
        sens = pupil_sensitivity_matrix(opt_tlt_coeffs,dtlt,Nmodes=100,alpha=alpha,flip=flip)
        synim = get_synim(Nmodes=100,flip=flip,alpha=alpha)
        G = np.diag(np.linalg.pinv(synim) @ refim[:,:100])
        aux = ((refim[:,:100] @ np.diag(1/G)) - synim)
        dtlt_coeffs = np.linalg.pinv(sens) @ aux.flatten()
        opt_tlt_coeffs += dtlt_coeffs.reshape([2,4])
        write_overrides(Nmodes,opt_tlt_coeffs)

        # Step 2: optimize rotation and shift
        sens = sensitivity_matrix(alpha,eps_vec=eps,Nmodes=Nmodes,flip=flip) 

        # Update gain and synIM
        synim = get_synim(Nmodes,flip=flip,alpha=alpha)
        G = np.diag(np.linalg.pinv(synim) @ refim[:,:Nmodes])

        # Update alpha
        aux = ((refim[:,:Nmodes] @ np.diag(1/G)) - synim)
        metric = np.sqrt(np.sum(aux**2))
        dalpha = np.linalg.pinv(sens) @ aux.flatten()
        print(f'Update parameters are: {dalpha}')
        alpha_new = alpha.copy()
        alpha_new[:len(dalpha)] += dalpha
        # err = np.max(np.abs(dalpha)/np.abs(alpha_new))
        # err = np.max(np.abs(dalpha)-np.abs(eps)/2)
        err = np.max(np.abs(dalpha)/np.abs(alpha_new[:len(dalpha)]))
        print(err,metric)

        # Update parameters
        eps = eps/2 + eps/2*(np.abs(dalpha)>eps)
        alpha = alpha_new
        k += 1
    
    if k == max_its:
        print(f'\nOptimization did not converge in {max_its} iterations! Last parameters: {alpha}')
    else:
        print(f'\nOptimization success in {k} iterations! Found parameters: {alpha}')
        save_ifunc_pars(flip=flip,rot=alpha[0],shiftX=alpha[1],shiftY=alpha[2],mag=alpha[3],scaleX=alpha[4],scaleY=alpha[5],phi=alpha[0])

                


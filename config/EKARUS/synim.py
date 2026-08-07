from astropy.io import fits
import numpy as np
import os
import pandas as pd

import specula 
specula.init(0)

from skimage.transform import AffineTransform,warp
from scipy.ndimage import rotate

from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.mmlib.utils import get_pupil_mask, get_frame_pupil_centers
from specula.mmlib.yaml_overrides import write_yaml_overrides

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
    img = np.zeros(ekapup.shape)
    center_y, center_x = img.shape[0]/2.0, img.shape[1]/ 2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    shear_and_scale = AffineTransform(shear=shear, rotation=rot, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x, center_y))
    trf = shift_to_origin + shear_and_scale + shift_to_center
    # warp_mask = np.logical_or(warp((1-ekapup), inverse_map=trf.inverse),(1-ekapup).astype(bool))
    for j in range(ifunc.shape[1]):
        img[ekapup.astype(bool)] = ifunc[:,j]
        warp_img = warp(img, inverse_map=trf.inverse)
        ifunc_new[:,j] = warp_img[ekapup.astype(bool)]
    return ifunc_new

def rotate_ifunc(ifunc,rot_deg:float):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(ekapup.shape)
    for j in range(ifunc.shape[1]):
        img[ekapup.astype(bool)] = ifunc[:,j]
        rot_img = rotate(img,angle=rot_deg,reshape=False)
        ifunc_new[:,j] = rot_img[ekapup.astype(bool)]
    return ifunc_new

def shift_ifunc(ifunc,shift:float,ax_dir):
    ifunc_new = np.zeros_like(ifunc)
    img = np.zeros(ekapup.shape)
    for j in range(ifunc.shape[1]):
        img[ekapup.astype(bool)] = ifunc[:,j]
        shift_img = shift_image(img,shift=shift,axis=ax_dir)
        ifunc_new[:,j] = shift_img[ekapup.astype(bool)]
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
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=ekapup)
    ifunc_obj.save(f'/raid1/mmenessini/calibration/EKARUS/ifunc/{ifunc_tag}.fits', overwrite=True)

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
        ifunc_inv_new[:] = warp_image(ifunc_inv_new.T,mag=mag).T
    if shearAmp is not None:
        ifunc_new[:] = warp_image(ifunc_new,shear=shearAmp,rot=shearAngle)
        ifunc_inv_new[:] = warp_image(ifunc_inv_new.T,shear=shearAmp,rot=shearAngle).T
    ifunc_obj = IFunc(ifunc=ifunc_new.T,mask=ekapup)
    ifunc_obj.save('/raid1/mmenessini/calibration/EKARUS/ifunc/dm468_ifunc_bestshift.fits', overwrite=True)
    ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv_new.T,mask=ekapup)
    ifunc_inv_obj.save('/raid1/mmenessini/calibration/EKARUS/ifunc/dm468_ifunc_bestshift_inv.fits', overwrite=True)

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

def evaluate_error(Nmodes:int):
    calibim = fits.getdata(f'/raid1/mmenessini/calibration/EKARUS/im/{im_tag}.fits')[:,:Nmodes]
    calibim -= np.mean(calibim,axis=0)
    calibim *= np.std(refim,axis=0)/np.std(calibim,axis=0)
    err = np.zeros(Nmodes)
    for j in range(Nmodes):
        want = refim[:,j]
        fimg = np.zeros([240,240])
        rtot = 0
        for i in range(4):
            valid_ids = pup_ids[:,i] != -1
            len_ids = np.sum(valid_ids)
            np.put(fimg, pup_ids[valid_ids,i], calibim[rtot:rtot+len_ids,j])
            rtot += len_ids
        f2d = fimg.reshape([240,240])[60:180,60:180]
        got = f2d[crop_pyr_mask]
        delta = want - got
        err[j] = np.sqrt(np.sum(delta**2))
    return err

def evaluate_metric(Nmodes,return_err:bool=False):
    ovdes = ("{"
            f"main.total_time: {Nmodes*0.001*2}, "
            f"dm.nmodes: {Nmodes}, "
            f"pushpull.nmodes: {Nmodes}, "  
            f"pushpull.amp:    3000000, "
            f"pyr_im_calibrator.nmodes: {Nmodes}, "
            f"pyr_im_calibrator.im_tag: {im_tag}, "
            f"pyr_im_calibrator.overwrite: true, "
            f"pyr.mod_amp: {rMod:1.1f}, "
            f"dm.ifunc_object:      {ifunc_tag}, "
            f"dm.m2c_object:        {m2c_tag}, "
            "}")
    write_yaml_overrides(input_string=ovdes, temp_name='temp_synim')
    main_config = 'ekarus_onbench.yml calib_im.yml'
    os.system(f"specula {main_config} temp_synim.yml")
    err = evaluate_error(Nmodes)
    metric = np.sqrt(np.sum(err**2))
    if return_err:
        return metric,err
    else:
        return metric


delta_vec = lambda vec: (np.max(vec)-np.min(vec))/len(vec)


if __name__ == "__main__":

    Nmodes = 400

    rot0 = 0.0
    shiftX0 = 0.0
    shiftY0 = 0.00
    mag0 = 1.00
    # rot0 = 0.33
    # shiftX0 = 0.35
    # shiftY0 = -0.02
    # mag0 = 1.013
    shearAmp0 = 0
    shearAngle0 = 0

    prefix = 'oldIM_it0_'
    overwrite = False

    rotvec = np.linspace(-5.0,5.0,21)
    shiftvec = np.linspace(-1.0,1.0,21)
    dmags = np.linspace(-0.03,0.05,15)
    # rotvec = np.linspace(-0.25,0.25,11)
    # shiftvec = np.linspace(-0.2,0.2,11)
    # dmags = np.linspace(-0.005,0.005,11)

    shear_amps = np.linspace(-0.05,0.05,5)
    shear_angles = np.linspace(-np.pi,np.pi,36)

    result_dir = '/raid1/mmenessini/results/EKARUS/'

    columns = ['rotation','shiftX','shiftY','magnification','shearAmp','shearAngle','metric']
    result = []

    save_ifunc_pars(ifunc,rot=rot0,shiftX=shiftX0,shiftY=shiftY0,mag=mag0)
    # print(ifunc.shape,ekapup.shape)

    err_rot = np.zeros([len(rotvec),Nmodes])
    for j,rot in enumerate(rotvec):
        print(f'Testing rotation: {rot}')
        set_ifunc_pars(ifunc,rot=rot0+rot,shiftX=shiftX0,shiftY=shiftY0,shearAmp=shearAmp0,shearAngle=shearAngle0)
        chi,err_rot[j] = evaluate_metric(Nmodes,return_err=True)
        result.append({'rotation': rot+rot0, 'shiftX': shiftX0, 'shiftY': shiftY0, 'shearAmp': shearAmp0, 'shearAngle': shearAngle0, 'magnification': mag0, 'metric': chi})
        print(f'Obtained metric: {chi}')
    fits.writeto(os.path.join(result_dir, 'misreg_csv',prefix+f'deltaRot{(np.max(rotvec)-np.min(rotvec))/len(rotvec):1.2f}_{Nmodes}modes_metrics.fits'),err_rot,overwrite=overwrite)

    err_shft = np.zeros([len(shiftvec),Nmodes])
    for j,shft in enumerate(shiftvec):
        print(f'Testing x-shift: {shft}')
        set_ifunc_pars(ifunc,rot=rot0,shiftX=shft+shiftX0,shiftY=shiftY0,shearAmp=shearAmp0,shearAngle=shearAngle0)
        chi,err_shft[j] = evaluate_metric(Nmodes,return_err=True)
        result.append({'rotation': rot0, 'shiftX': shft+shiftX0, 'shiftY': shiftY0, 'shearAmp': shearAmp0, 'shearAngle': shearAngle0,  'magnification': mag0, 'metric': chi})
        print(f'Obtained metric: {chi}')
    fits.writeto(os.path.join(result_dir, 'misreg_csv',prefix+f'deltaShiftX{(np.max(shiftvec)-np.min(shiftvec))/len(shiftvec):1.2f}_{Nmodes}modes_metrics.fits'),err_shft,overwrite=overwrite)

    for j,shft in enumerate(shiftvec):
        print(f'Testing y-shift: {shft}')
        set_ifunc_pars(ifunc,rot=rot0,shiftY=shft+shiftY0,shiftX=shiftX0,shearAmp=shearAmp0,shearAngle=shearAngle0,mag=mag0)
        chi,err_shft[j] = evaluate_metric(Nmodes,return_err=True)
        result.append({'rotation': rot0, 'shiftX': shiftX0, 'shiftY': shft+shiftY0, 'shearAmp': shearAmp0, 'shearAngle': shearAngle0,  'magnification': mag0, 'metric': chi})
        print(f'Obtained metric: {chi}')
    fits.writeto(os.path.join(result_dir, 'misreg_csv',prefix+f'deltaShiftY{(np.max(shiftvec)-np.min(shiftvec))/len(shiftvec):1.2f}_{Nmodes}modes_metrics.fits'),err_shft,overwrite=overwrite)

    err_mag = np.zeros([len(dmags),Nmodes])
    for j,dmag in enumerate(dmags):
        print(f'Testing magnification: {(mag0+dmag)*1e+2:1.1f}%')
        set_ifunc_pars(ifunc,rot=rot0,shiftY=shiftY0,shiftX=shiftX0,mag=mag0+dmag,shearAmp=shearAmp0,shearAngle=shearAngle0)
        chi,err_mag[j] = evaluate_metric(Nmodes,return_err=True)
        result.append({'rotation': rot0, 'shiftX': shiftX0, 'shiftY': shiftY0, 'shearAmp': shearAmp0, 'shearAngle': shearAngle0,  'magnification': mag0+dmag, 'metric': chi})
        print(f'Obtained metric: {chi}')
    fits.writeto(os.path.join(result_dir, 'misreg_csv',prefix+f'deltaMag{1e+2*(np.max(dmags)-np.min(dmags))/len(dmags):1.2f}_{Nmodes}modes_metrics.fits'),err_mag,overwrite=overwrite)

    # err_mag = np.zeros([len(shear_amps),len(shear_angles),Nmodes])
    # for i,shear in enumerate(shear_amps):
    #     for j,angle in enumerate(shear_angles):
    #         print(f'Testing shear {shear} with angle: {angle*180/np.pi}°')
    #         set_ifunc_pars(ifunc,rot=rot0,shiftY=shiftY0,shiftX=shiftX0,shearAngle=shearAngle0+angle,shearAmp=shearAmp0+shear)
    #         chi,err_mag[i,j] = evaluate_metric(Nmodes,return_err=True)
    #         result.append({'rotation': rot0, 'shiftX': shiftX0, 'shiftY': shiftY0,  'shearAmp': shearAmp0+shear, 'shearAngle': shearAngle0+angle, 'magnification': mag0, 'metric': chi})
    #         print(f'Obtained metric: {chi}')
    # fits.writeto(os.path.join(result_dir, 'misreg_csv', prefix+f'deltaShear{delta_vec(shear_amps):1.2f}dAngle{delta_vec(shear_angles)*180/np.pi:1.0f}deg_{Nmodes}modes_metrics.fits'),err_mag,overwrite=overwrite)
    results_df = pd.DataFrame(result, columns=columns) 
    results_df.to_csv(os.path.join(result_dir, 'misreg_csv', 
                                   prefix+f'deltaRot{delta_vec(rotvec):1.2f}_deltaShift{delta_vec(shiftvec):1.2f}_deltaMag{1e+2*delta_vec(dmags):1.2f}_shear{delta_vec(shear_amps):1.2f}dAngle{delta_vec(shear_angles)*180/np.pi:1.0f}_metric.csv'), index=False)
        


                


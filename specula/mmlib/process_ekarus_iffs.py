import specula
specula.init(0)  # Use GPU device 0 (or -1 for CPU)

# import numpy as np
import os

from astropy.io import fits

from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.m2c import M2C
from specula.calib_manager import CalibManager
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.lib.toccd import toccd

# from specula import cpuArray


def regularize_mat(mat, thr:float=1e-2):
    U,S,Vt = specula.xp.linalg.svd(mat,full_matrices=False)
    Sreg = S+S.max()*thr #np.maximum(S,S.max()*thr)
    mat = (U * Sreg) @ Vt
    print(f'Regularized {specula.xp.sum(S<thr*S.max()):1.0f} eigenvalues')
    return mat

def postprocess_iffs(root_dir:str, data_path:str, tag:str, mask_tag:str, Npix:int, D:float, r0=0.03, L0=25):
    # iffs = specula.xp.array(fits.getdata(os.path.join(data_path,'DM468_IFFs.fits')))
    # mask = specula.xp.array(fits.getdata(os.path.join(data_path,'DM468_mask.fits')),dtype=bool)
    # iffs = specula.xp.array(fits.getdata(os.path.join(data_path,'alpaoIFFs.fits')))
    # mask = specula.xp.array(1-fits.getdata(os.path.join(data_path,'alpaoPupMask.fits')),dtype=bool)
    iffs = specula.xp.array(fits.getdata(os.path.join(data_path,'reordered_IFs.fits')))
    mask = specula.xp.array(fits.getdata(os.path.join(data_path,'IFmask.fits')),dtype=bool)
    iffs = specula.xp.array(fits.getdata(os.path.join(data_path,'purged_trim_IFs.fits')))
    mask = specula.xp.array(1-fits.getdata(os.path.join(data_path,'trim_IFmask.fits')),dtype=bool)

    X,Y = specula.xp.mgrid[0:mask.shape[0],0:mask.shape[1]]
    minX = specula.xp.min(X[~mask.astype(bool)])
    maxX = specula.xp.max(X[~mask.astype(bool)])
    minY = specula.xp.min(Y[~mask.astype(bool)])
    maxY = specula.xp.max(Y[~mask.astype(bool)])

    crop_mask = mask[int(minX-1):int(maxX+1),:]
    crop_mask = crop_mask[:,int(minY-1):int(maxY+1)]

    # print(specula.xp.sum(1-mask),specula.xp.sum(1-crop_mask),mask.shape,mask.dtype,minX,maxX,minY,maxY)

    Nacts = specula.xp.shape(iffs)[0]
    aux = specula.xp.zeros(crop_mask.shape,dtype=specula.xp.float32)

    # binned_shape = (Npix,Npix)
    # bin_mask = toccd(crop_mask.astype(float), binned_shape, xp=specula.xp) > 0
    bin_mask = crop_mask.copy()
    binned_shape = crop_mask.shape
    Npix = binned_shape[0]
    # print(bin_mask.shape,specula.xp.sum(1-bin_mask),specula.xp.sum(bin_mask))
    # fits.writeto(os.path.join(root_dir,'mask','bin_mask.fits'),cpuArray(bin_mask.astype(float)),overwrite=True)
    # pupil = specula.xp.array(fits.getdata(os.path.join(root_dir,'pupilstop',mask_tag+'.fits')),dtype=bool)
    # print(pupil.shape,specula.xp.sum(1-pupil),specula.xp.sum(pupil))
    # bin_pup_mask = specula.xp.logical_or(bin_mask,1-pupil)
    # fits.writeto(os.path.join(root_dir,'mask','pup_mask.fits'),cpuArray(bin_mask.astype(float)),overwrite=True)
    # print(bin_mask.shape,specula.xp.sum(1-bin_mask))


    # # Regularized influence functions
    # iffs = regularize_mat(iffs, thr=1e-2)  # was 2e/4

    IF = specula.xp.zeros([Nacts,int(specula.xp.sum(1-bin_mask))]) # bin_pup_mask
    unobsIF = specula.xp.zeros([Nacts,int(specula.xp.sum(1-bin_mask))])

    for j in range(Nacts):
        aux[~crop_mask] = iffs[j,:]
        bin_if = toccd(aux,binned_shape,xp=specula.xp)
        IF[j,:] = bin_if[~bin_mask]
        unobsIF[j,:] = bin_if[~bin_mask]

    
    # # Regularized influence functions
    # IF = regularize_mat(IF, thr=1e-3)

    zern_modes = 2
    oversampling = 4
    
    # kl_basis, m2c, _ = make_modal_base_from_ifs_fft(
    #     pupil_mask=specula.xp.array(1-bin_pup_mask),
    #     diameter=D,
    #     influence_functions=IF,
    #     r0=r0,
    #     L0=L0,
    #     zern_modes=zern_modes,
    #     oversampling=oversampling,
    #     if_max_condition_number=1e+3, #None, #1e+4,
    #     xp=specula.xp,
    #     dtype=specula.xp.float32
    # )

    kl_basis, m2c, _ = make_modal_base_from_ifs_fft(
        pupil_mask=specula.xp.array(1-bin_mask),
        diameter=D,
        influence_functions=unobsIF,
        r0=r0,
        L0=L0,
        zern_modes=zern_modes,
        oversampling=oversampling,
        if_max_condition_number=None,
        xp=specula.xp,
        dtype=specula.xp.float32
    )

    # aux = specula.xp.zeros(bin_mask.shape,dtype=specula.xp.float32)
    # kl_basis_copy = kl_basis.copy()
    # kl_basis = specula.xp.zeros([kl_basis.shape[0],int(specula.xp.sum(1-bin_pup_mask))])
    # for j in range(kl_basis.shape[0]):
    #     aux[~bin_mask] = kl_basis_copy[j,:]
    #     kl_basis[j,:] = aux[~bin_pup_mask]

    # kl_basis = regularize_mat(kl_basis, thr=1e-2)
    # # m2c = regularize_mat(m2c, thr=1e-2)

    kl_basis_inv = specula.xp.linalg.pinv(kl_basis)

    ##########################################################    
    os.makedirs(root_dir, exist_ok=True)

    # initialize calibration manager
    calib_manager = CalibManager(root_dir)

    # tags
    ifunc_tag = tag+'_ifunc'
    m2c_tag = tag+'_m2c'
    base_inv_tag = tag+'_kl_inv'

    ifunc_filename = calib_manager.filename('ifunc', ifunc_tag)
    m2c_filename = calib_manager.filename('m2c', m2c_tag)
    base_inv_filename = calib_manager.filename('ifunc', base_inv_tag)

    print(f"Influence functions shape: {IF.shape}")
    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    # Step 3: Create output directory
    os.makedirs(os.path.join(root_dir, 'pupilstop'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=IF,
        mask=(1-bin_mask).astype(specula.xp.uint8) # bin_pup_mask
    )
    ifunc_obj.save(ifunc_filename, overwrite=True)
    print("OK: " + ifunc_filename + f" (zonal influence functions {IF.shape})")

    # Create M2C object for mode-to-command matrix and save
    m2c_obj = M2C(
        m2c=m2c
    )
    m2c_obj.save(m2c_filename, overwrite=True)
    print("OK: " + m2c_filename + f" (m2c matrix: {m2c.shape})")

    # inverse influence function object for modal analysis
    print(f"\nSaving inverse modal base...")
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_basis_inv,
        mask=(1-bin_mask).astype(specula.xp.uint8) # bin_pup_mask
    )
    ifunc_inv_obj.save(base_inv_filename, overwrite=True)
    print("OK: " + base_inv_filename + f" (inverse modal base: {kl_basis_inv.shape})")

    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupil_mask = Pupilstop(simul_params=simul_params, input_mask=1-bin_mask) # bin_pup_mask
    fname = os.path.join(root_dir, 'pupilstop', tag+f'_{Npix:1.0f}pixels.fits')
    pupil_mask.save(fname)
    print("OK: " + fname + f" (pupil mask: {bin_mask.shape})")



if __name__ == "__main__":
    D = 1.82
    data_path = '/raid1/mmenessini/calibration/EKARUS/data'
    root_dir = '/raid1/mmenessini/calibration/EKARUS'

    postprocess_iffs(root_dir=root_dir, data_path=data_path, mask_tag='copernico_pupil_120pixels', tag='purged_trim_DM468', Npix=120, D=D)




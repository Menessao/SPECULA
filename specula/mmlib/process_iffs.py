import specula
specula.init(-1)  # Use GPU device 0 (or -1 for CPU)

import numpy as np
import os

from scipy.io import readsav

from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.m2c import M2C
from specula.calib_manager import CalibManager
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft

def postprocess_iffs(root_dir:str, data_path:str, tag:str, D:float, r0=0.1, L0=25):
    fname = os.path.join(data_path,'phase_matrix.sav')
    data_dict = readsav(fname)

    kl_basis = data_dict['klmatrix'].T
    dpix = data_dict['dpix']
    m2c_full = data_dict['klm2c']
    mode_ids = (np.sum(abs(m2c_full),axis=0)>0).astype(bool)
    act_ids = (np.sum(abs(m2c_full),axis=1)>0).astype(bool)
    m2c = m2c_full[:,mode_ids]
    m2c = m2c[act_ids,:]
    idx = data_dict['idx_mask']

    # print(m2c_full.shape,np.sum(np.sum(abs(m2c_full),axis=0)>0), np.sum(np.sum(abs(m2c_full),axis=1)>0))

    print('Reading data from: '+fname)
    print(f'Scale: {dpix} pixels across diameter')

    # Pupil mask
    pmask = specula.xp.ones([dpix,dpix],dtype=bool).flatten()
    pmask[idx] = False
    pmask = pmask.reshape([dpix,dpix])
    simul_params = SimulParams(pixel_pupil=dpix,pixel_pitch=D/dpix)
    pupil_mask = Pupilstop(simul_params=simul_params, input_mask=1-pmask)

    # Influence functions
    influence_functions = kl_basis.T @ specula.xp.linalg.pinv(m2c)

    mask_pixels = np.sum(1-pmask)
    pupil_pixels = influence_functions.shape[0]
    print(f"Valid mask pixels: {mask_pixels}")
    print(f"Pupil pixels: {pupil_pixels}")

    ##########################################################    
    os.makedirs(root_dir, exist_ok=True)

    # initialize calibration manager
    calib_manager = CalibManager(root_dir)

    # tags
    ifunc_tag = tag+'_ifunc'
    m2c_tag = tag+'_m2c'
    m2c_full_tag = tag+'_m2c_full'
    base_inv_tag = tag+'_kl_inv'

    ifunc_filename = calib_manager.filename('ifunc', ifunc_tag)
    m2c_filename = calib_manager.filename('m2c', m2c_tag)
    m2c_full_filename = calib_manager.filename('m2c', m2c_full_tag)
    base_inv_filename = calib_manager.filename('ifunc', base_inv_tag)

    # Regularized influence functions
    thr = 1e-2
    U,S,Vt = np.linalg.svd(influence_functions,full_matrices=False)
    Sreg = S+S.max()*thr #np.maximum(S,S.max()*thr)
    influence_functions = (U * Sreg) @ Vt
    print(f'Regularized {np.sum(S<thr*S.max()):1.0f} eigenvalues')

    print(f"Influence functions shape: {influence_functions.shape}")
    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    zern_modes = 2
    oversampling = 4

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=specula.xp.array(1-pmask), #specula.xp.array(unobs_pupil_mask),#
        diameter=D,
        influence_functions=influence_functions.T,
        r0=r0,
        L0=L0,
        zern_modes=zern_modes,
        oversampling=oversampling,
        if_max_condition_number=1e+2,
        xp=specula.xp,
        dtype=specula.xp.float32
    )

    kl_basis_inv = np.linalg.pinv(kl_basis)

    # Step 3: Create output directory
    os.makedirs(os.path.join(root_dir, 'pupilstop'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=influence_functions.T,
        mask=(1-pmask).astype(np.uint8)
    )
    ifunc_obj.save(ifunc_filename, overwrite=True)
    print("OK: " + ifunc_filename + f" (zonal influence functions {influence_functions.T.shape})")

    # Create M2C object for mode-to-command matrix and save
    m2c_obj = M2C(
        m2c=m2c
    )
    m2c_obj.save(m2c_filename, overwrite=True)
    print("OK: " + m2c_filename + f" (m2c matrix: {m2c.shape})")
    
    m2c_obj = M2C(
        m2c=m2c_full
    )
    m2c_obj.save(m2c_full_filename, overwrite=True)
    print("OK: " + m2c_full_filename + f" (m2c matrix: {m2c_full.shape})")

    # inverse influence function object for modal analysis
    print(f"\nSaving inverse modal base...")
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_basis_inv,
        mask=(1-pmask).astype(np.uint8)
    )
    ifunc_inv_obj.save(base_inv_filename, overwrite=True)
    print("OK: " + base_inv_filename + f" (inverse modal base: {kl_basis_inv.shape})")

    # Save actual base inv
    kl = influence_functions @ m2c
    kl_inv = np.linalg.pinv(kl.T)
    base_inv_filename = calib_manager.filename('ifunc', tag+'_ifunc_inv')
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_inv,
        mask=(1-pmask).astype(np.uint8)
    )
    ifunc_inv_obj.save(base_inv_filename, overwrite=True)
    print("OK: " + base_inv_filename + f" (inverse modal base: {kl_inv.shape})")

    fname = os.path.join(root_dir, 'pupilstop', tag+f'_{dpix:1.0f}pixels.fits')
    pupil_mask.save(fname)






if __name__ == "__main__":
    D = 8.4
    data_path = '/raid1/mmenessini/LBTData/KLv30dx'
    soul_dir = '/raid1/mmenessini/calibration/SOUL/KLv30dx'

    postprocess_iffs(root_dir=soul_dir, data_path=data_path, tag='asm_v30dx', D=D)




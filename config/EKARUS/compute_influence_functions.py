import specula
specula.init(0)  # Use GPU device 0 (or -1 for CPU)

import numpy as np
import os
from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.data_objects.m2c import M2C
from specula.calib_manager import CalibManager
from specula import cpuArray

def compute_and_save_influence_functions(r0=0.15,L0=25,n_acts=24,zern_modes:int=5,geom='alpao',overwrite=True):
    """
    Compute zonal influence functions and modal basis for the SCAO tutorial
    Follows the same approach as test_modal_basis.py
    """
    # create calibration directory if it doesn't exist
    root_dir = './calib'
    os.makedirs(root_dir, exist_ok=True)

    # tags
    ifunc_tag = 'scao_pyr_ifunc'
    m2c_tag = 'scao_pyr_m2c'
    modal_base_tag = 'scao_pyr_mbase'
    base_inv_tag = 'scao_pyr_mbase_inv'

    # initialize calibration manager
    calib_manager = CalibManager(root_dir)

    # DM and pupil parameters for VLT-like telescope
    pupil_pixels = 160           # Pupil sampling resolution
    n_actuators = n_acts
    telescope_diameter = 1.82     # meters (Copernico telescope)

    # Pupil geometry
    obsratio = 0.33               # 33% central obstruction
    diaratio = 1.0               # Full pupil diameter

    # Actuator geometry - aligned with test_modal_basis.py
    angleOffset = 0              # No rotation

    # Actuator slaving (disable edge actuators outside pupil)
    doSlaving = True             # Enable slaving (very simple slaving)
    slavingThr = 0.1             # Threshold for master actuators

    # Modal basis parameters
    oversampling = 2             # No oversampling

    # Computation parameters
    dtype = specula.xp.float32   # Use current device precision

    print("Computing zonal influence functions...")
    print(f"Pupil pixels: {pupil_pixels}")
    print(f"Telescope diameter: {telescope_diameter}m")
    print(f"Central obstruction: {obsratio*100:.1f}%")
    print(f"r0 = {r0}m, L0 = {L0}m")

    # Step 1: Generate zonal influence functions
    influence_functions, pupil_mask = compute_zonal_ifunc(
        pupil_pixels,
        n_actuators,
        geom=geom,
        angle_offset=angleOffset,
        do_slaving=doSlaving,
        slaving_thr=slavingThr,
        obsratio=obsratio,
        diaratio=diaratio,
        mask=None,
        xp=specula.xp,
        dtype=dtype,
        return_coordinates=False
    )

    # Print statistics
    n_valid_actuators = influence_functions.shape[0]
    n_pupil_pixels = specula.xp.sum(pupil_mask)

    print(f"\nZonal influence functions:")
    print(f"Valid actuators: {n_valid_actuators}/{n_actuators**2} ({n_valid_actuators/(n_actuators**2)*100:.1f}%)")
    print(f"Pupil pixels: {int(n_pupil_pixels)}/{pupil_pixels**2} ({float(n_pupil_pixels)/(pupil_pixels**2)*100:.1f}%)")
    print(f"Influence functions shape: {influence_functions.shape}")

    # Step 2: Generate modal basis (KL modes)
    print(f"\nGenerating KL modal basis...")

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=pupil_mask,
        diameter=telescope_diameter,
        influence_functions=influence_functions,
        r0=r0,
        L0=L0,
        zern_modes=zern_modes,
        oversampling=oversampling,
        if_max_condition_number=None,
        xp=specula.xp,
        dtype=dtype
    )

    print(f"KL basis shape: {kl_basis.shape}")
    print(f"Number of KL modes: {kl_basis.shape[0]}")

    kl_basis_inv = np.linalg.pinv(kl_basis)

    # Step 3: Create output directory
    os.makedirs(os.path.join(root_dir, 'ifunc'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'm2c'), exist_ok=True)

    # Step 4: Save using SPECULA data objects
    print(f"\nSaving influence functions and modal basis...")

    # Create IFunc object and save
    ifunc_obj = IFunc(
        ifunc=influence_functions,
        mask=pupil_mask
    )
    ifunc_filename = calib_manager.filename('ifunc', ifunc_tag)
    ifunc_obj.save(ifunc_filename,overwrite=overwrite)
    print("OK: " + ifunc_filename + " (zonal influence functions)")

    # Create M2C object for mode-to-command matrix and save
    m2c_obj = M2C(
        m2c=m2c
    )
    m2c_filename = calib_manager.filename('m2c', m2c_tag)
    m2c_obj.save(m2c_filename,overwrite=overwrite)
    print("OK: " + m2c_filename + " (m2c)")

    # modal base
    print(f"\nSaving inverse modal base...")
    modal_base_obj = IFunc(
        ifunc=kl_basis,
        mask=pupil_mask
    )
    modal_base_filename = calib_manager.filename('ifunc', modal_base_tag)
    modal_base_obj.save(modal_base_filename,overwrite=overwrite)
    print("OK: " + modal_base_filename + " (modal base)")

    # inverse influence function object for modal analysis
    print(f"\nSaving inverse modal base...")
    ifunc_inv_obj = IFuncInv(
        ifunc_inv=kl_basis_inv,
        mask=pupil_mask
    )
    base_inv_filename = calib_manager.filename('ifunc', base_inv_tag)
    ifunc_inv_obj.save(base_inv_filename,overwrite=overwrite)
    print("OK: " + base_inv_filename + " (inverse modal base)")

    # Step 5: Optional visualization
    try:
      import matplotlib.pyplot as plt

      print(f"\nGenerating visualization...")

      plt.figure(figsize=(10, 6))
      plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF Covariance')
      plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence Covariance')
      plt.xlabel('Mode number')
      plt.ylabel('Singular value')
      plt.title('Singular values of covariance matrices')
      plt.legend()
      plt.grid(True)

      # move to CPU / numpy for plotting if required
      kl_basis = cpuArray(kl_basis)
      pupil_mask = cpuArray(pupil_mask)

      # Plot some modes
      max_modes = min(36, kl_basis.shape[0])

      # Create a mask array for display
      mode_display = np.zeros((max_modes, pupil_mask.shape[0], pupil_mask.shape[1]))
      mode_ids = np.zeros(max_modes,dtype=int)

      # Place each mode vector into the 2D pupil shape
      idx_mask = np.where(pupil_mask)
      for i in range(max_modes//2):
          mode_img = np.zeros(pupil_mask.shape)
          mode_img[idx_mask] = kl_basis[i]
          mode_display[i] = mode_img
          mode_ids[i] = i+1
          mode_img[idx_mask] = kl_basis[-(max_modes//2-i+1)]
          mode_display[i+max_modes//2] = mode_img
          mode_ids[i+max_modes//2] = kl_basis.shape[0]-max_modes//2+i


      # Plot the reshaped modes
      n_rows = int(np.round(np.sqrt(max_modes)))
      n_cols = int(np.ceil(max_modes / n_rows))
      plt.figure(figsize=(18, 12))
      for i in range(max_modes):
          plt.subplot(n_rows, n_cols, i+1)
          plt.imshow(np.ma.masked_array(mode_display[i],mask=1-pupil_mask), cmap='viridis',origin='lower')
          plt.title(f'Mode {mode_ids[i]+1}')
          plt.axis('off')
      plt.tight_layout()

      plt.show()

    except ImportError:
        print("Matplotlib not available - skipping visualization")

    print(f"\nInfluence functions and modal basis computation completed!")
    print(f"Files saved using CalibManager in: {calib_manager.root_dir}")
    print(f"\nFiles created:")
    print(f"  {ifunc_filename}        - Zonal influence functions ({n_valid_actuators} actuators)")
    print(f"  {m2c_filename}          - m2c {m2c.shape}")
    print(f"  {base_inv_filename}     - Inverse modal base")

    # Step 6: Test loading the saved files
    print(f"\nTesting file loading...")

    try:
        # Test IFunc loading
        loaded_ifunc = IFunc.restore(ifunc_filename)
        assert loaded_ifunc.influence_function.shape == influence_functions.shape
        print("OK: IFunc loading test passed")

        # Test M2C loading
        loaded_m2c = M2C.restore(m2c_filename)
        print('Shapes: ', loaded_m2c.m2c.shape, kl_basis.shape)
        assert loaded_m2c.m2c.shape == m2c.shape # kl_basis.shape
        print("OK: M2C loading test passed")

    except Exception as e:
        print(f"⚠ File loading test failed: {e}")

    return ifunc_obj, m2c_obj

if __name__ == "__main__":
    compute_and_save_influence_functions()

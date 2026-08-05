import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import welch

import specula
specula.init(0)  # Default target device

from specula.lib.calc_psf import calc_psf
from specula.lib.make_mask import make_mask
from specula.data_objects.iir_filter_data import IirFilterData

def r0_from_seeing(seeing):
    return 0.98*500e-9/seeing*180/np.pi*3600

def seeing_from_r0(r0):
    return 0.98*500e-9/r0*180/np.pi*3600

def radial_order(i_mode, xp=np):
    noll = i_mode + 2
    return (xp.ceil(-3.0/2.0+xp.sqrt(1+8*noll)/2.0)).astype(int)

def compute_modal_variance_von_karman(r0, L0, D, modes, mask):
    """
    Computes the turbulent power projected on a given modal basis 
    assuming a Von Karman phase power spectrum.
    
    Parameters:
    - r0: float, Fried parameter [m]
    - L0: float, Outer scale [m]
    - D: float, Telescope diameter [m]
    - modes: 2D numpy array of shape (N_modes, N_valid_pixels) containing the modal basis
    - mask: 2D numpy array of shape (N_pixels, N_pixels) representing the telescope pupil
    
    Returns:
    - modal_variances: 1D numpy array of length N_modes containing the variance 
                       (turbulent power) of each mode [rad^2]
        WARNING: Must be divided by telescope_area**2 to obtain the correct units !
    """
    n_modes = modes.shape[0] 
    nx, ny = mask.shape
    
    # Assuming the grid spans the telescope diameter exactly
    # Pixel scale in the pupil plane [m/pixel]
    dx = D / nx  
    dy = D / ny
    
    # Generate spatial frequency grids [1/m]
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy)
    
    # Squared spatial frequency magnitude
    F2 = FX**2 + FY**2
    
    # Von Karman phase power spectrum (Phi(f) in [rad^2 * m^2])
    # Formula: Phi_VK(f) = 0.0229 * r0^(-5/3) * (f^2 + (1/L0)^2)^(-11/6)
    phi_vk = 0.0229 * (r0 ** (-5/3)) * (F2 + (1.0 / L0)**2) ** (-11/6)
    
    # Frequency grid step sizes for integration [1/m]
    dfx = 1.0 / (nx * dx)
    dfy = 1.0 / (ny * dy)
    
    modal_variances = np.zeros(n_modes)
    mode_masked = np.zeros_like(mask,dtype=float)
    
    for i in range(n_modes):
        # Apply the telescope pupil mask to the mode
        mode_masked[~mask] = modes[i,:]/np.std(modes[i,:])
        
        # Continuous Fourier Transform approximation of the mode
        # Mode is in spatial domain, multiplied by pixel area (dx*dy) to scale the FFT appropriately
        mode_fft = np.fft.fft2(mode_masked) * (dx * dy)
        
        # Power Spectral Density of the mode |M(f)|^2
        mode_psd = np.abs(mode_fft) ** 2
        
        # Integrate the multiplication of the mode PSD and the Von Karman PSD 
        # over the spatial frequency space: integral(|M(f)|^2 * Phi(f) dfx dfy)
        modal_variances[i] = np.sum(mode_psd * phi_vk) * (dfx * dfy)
        
    return modal_variances

def von_karman_power(k,r0,L0,D):
    C = 0.02289558710855519
    B = k**2 + (D/L0)**2
    return C * (r0/D)**(-5.0/3.0) * B**(-11.0/6.0)

def get_pupil_mask(npix:int, filepath:str='', pyr:bool=True, pupdiam=None, obsratio=0.0, xp=np):
    if pyr:
        np_size = (npix,npix)
        pup_hdu = fits.open(filepath)
        rad = pup_hdu[2].data
        pup_ids = pup_hdu[1].data
        wfs_mask = xp.zeros(np_size)
        for j in range(len(rad)):
            f = xp.zeros(npix**2)
            xp.put(f, pup_ids[:,j], 1)
            f2d = f.reshape(np_size)
            wfs_mask += f2d
    else:
        wfs_mask = make_mask(np_size=npix, diaratio = pupdiam/npix, obsratio=obsratio)
    return wfs_mask.astype(bool)


def get_psd(data, dt:float, nperseg:int=1024):
    f,psd=welch(data,fs=1/dt,nperseg=nperseg,scaling='density',axis=-1)
    return psd,f


def get_reference_psf(root_dir:str,pupil_tag:str='vlt_pupil_160pixels',nd:int=4, xp=np):
    hdu = fits.open(os.path.join(root_dir,'pupilstop',pupil_tag+'.fits'))
    pupil = hdu[1].data
    psf = calc_psf(xp.zeros_like(pupil),pupil.astype(float), imwidth=int(pupil.shape[1]*nd), normalize=True,
                                            xp=np, complex_dtype=xp.complex128)
    return psf

def read_freq(params_path:str, obj_name:str=None):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
        if obj_name is None:
            fs = 1.0/float(params['main']['time_step'])
        else:
            fs = 1.0/float(params[obj_name]['dt'])
    return fs

def get_control_data(calib_dir:str,controller_name:str, gain_mod_name:str, params, xp=np):
    delay_frames = 1.0 + float(params[controller_name]['delay'])
    if params[controller_name]['class'] == 'Integrator':
        gain = xp.array(params[controller_name]['int_gain'])
        try:
            ff = xp.array(params[controller_name]['ff'])
        except KeyError:
            ff = xp.array([1.0])
        filter_data_complex = IirFilterData.from_gain_and_ff(gain=gain.tolist(),ff=ff.tolist())
    else:
        try:
            gain = float(params[gain_mod_name]['scheduled_values'][-1][0])
        except:
            gain = 1.0
        iir_path = os.path.join(calib_dir,'filter',str(params[controller_name]['iir_filter_data_object'])+'.fits')
        filter_data_complex = IirFilterData.restore(iir_path)
        filter_data_complex.num *= gain
    return filter_data_complex, delay_frames



def show_psf(psf, oversampling:int=4, title:str='', ext=0.25, vmin=-8, vmax=0, cmap='inferno', maxVal=None, xp=np):
    imageHalfSizeInPoints= psf.shape[0]/2
    roi= [int(imageHalfSizeInPoints*(1-ext)), int(imageHalfSizeInPoints*(1+ext))]
    psfZoom = psf[roi[0]: roi[1], roi[0]:roi[1]]
    sz = psfZoom.shape
    pixelSize = 1/oversampling
    if maxVal is None:
        maxVal = xp.max(psf)
    plt.imshow(xp.log10(psfZoom/maxVal), extent=
               [-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
               -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
               origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
    plt.xlabel(r'$\lambda/D$')
    plt.ylabel(r'$\lambda/D$')
    cbar= plt.colorbar()
    cbar.ax.set_title('Contrast')
    plt.title(title)
    

def reshape_on_mask(vec, mask, xp=np):
    """
    Reshape a given array on a 2D mask.
    :param flat_array: array of shape sum(1-mask)
    :param mask: boolean 2D mask
    :return: 2D array with flat_array in ~mask
    """
    image = xp.zeros(mask.shape, dtype=float)
    image[~mask] = vec
    image = xp.reshape(image, mask.shape)
    return xp.array(image)

    
def remap_on_new_mask(data, old_mask, new_mask, xp=np):
    """ 
    Remaps the matrix data defined on valid values 
    of old_mask to valid values on new_mask.

    Parameters
    ----------
    data : xp.ndarray
        2D array of shape (sum(1-old_mask), N)
    old_mask : xp.ndarray
        2D boolean array defining the old mask
    new_mask : xp.ndarray
        2D boolean array defining the new mask
    
    Returns
    -------
    remapped_data : xp.ndarray
        2D array of shape (sum(1-new_mask), N)
    """
    old_mask = xp.array(old_mask)
    new_mask = xp.array(new_mask)
    data = xp.array(data)

    old_len = xp.sum(1-old_mask)
    new_len = xp.sum(1-new_mask)

    if old_len < new_len:
        raise ValueError(f'Cannot reshape from {old_len} to {new_len}')

    transpose = False
    if xp.shape(data)[0] != old_len:
        data = data.T
        transpose = True

    if xp.shape(data)[0] != old_len:
        raise ValueError(f'Mask length {old_len} is incompatible with dimensions {data.shape}')
    elif len(xp.shape(data)) > 2:
        raise ValueError('Can only operate on 2D arrays')
    
    N = data.shape[1]
    remasked_data = xp.zeros([int(new_len),N])

    for j in range(N):
        old_data_2D = reshape_on_mask(data[:,j], old_mask, xp=xp)
        remasked_data[:,j] = old_data_2D[~new_mask]

    if transpose:
        remasked_data = remasked_data.T
    
    return remasked_data



def image_grid(shape, recenter:bool = False):
    ny, nx = shape
    cy, cx = (0,0)
    if recenter:
        cy, cx = ny//2, nx//2
    x = np.arange(nx, dtype=float) - cx
    y = np.arange(ny, dtype=float) - cy
    X,Y = np.meshgrid(x, y)
    return X,Y

def get_photocenter(image,offset:bool=False):
    X,Y = image_grid(image.shape)
    qy = np.sum(Y * image) / np.sum(image)
    qx = np.sum(X * image) / np.sum(image)
    if offset:
        qy += 0.5
        qx += 0.5
    return qx,qy 

def get_frame_pupil_centers(frame,thr=0.3,xhalf=100,yhalf=120):
    Y,X = image_grid(frame.shape)
    ll = (X<=xhalf) * (Y<=yhalf)
    lr = (X>xhalf) * (Y<=yhalf)
    ul = (X<=xhalf) * (Y>yhalf)
    ur = (X>xhalf) * (Y>yhalf)
    centers = np.zeros([4,2])
    mask = (frame*ll > thr*np.nanmax(frame*ll)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[0,:] = np.array([qx,qy])
    mask = (frame*lr > thr*np.nanmax(frame*lr)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[1,:] = np.array([qx,qy])
    mask = (frame*ul > thr*np.nanmax(frame*ul)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[2,:] = np.array([qx,qy])
    mask = (frame*ur > thr*np.nanmax(frame*ur)).astype(bool)
    qx,qy = get_photocenter(mask)
    centers[3,:] = np.array([qx,qy])
    return centers
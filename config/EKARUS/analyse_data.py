import os
import glob
import pickle
from astropy.io import fits
import matplotlib.pyplot as plt

import numpy as xp

def imageShow(image2d, pixelSize=1, title='', xlabel='', ylabel='', zlabel='', shrink=1.0):
    sz=image2d.shape
    plt.imshow(image2d, extent=[-sz[0]/2*pixelSize, sz[0]/2*pixelSize,
                                -sz[1]/2*pixelSize, sz[1]/2*pixelSize],
                                cmap = 'twilight')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar= plt.colorbar(shrink=shrink)
    cbar.ax.set_ylabel(zlabel)

def showZoomCenter(image, pixelSize, **kwargs):
    '''show log(image) zoomed around center'''
    imageHalfSizeInPoints= image.shape[0]/2
    roi= [int(imageHalfSizeInPoints*0.6), int(imageHalfSizeInPoints*1.4)]
    imageZoomedLog= xp.log(image[roi[0]: roi[1], roi[0]:roi[1]])
    imageShow(imageZoomedLog, pixelSize=pixelSize, **kwargs)



# Find all directories in ./output starting with '20'
dirs = [d for d in glob.glob("./output/20*") if os.path.isdir(d)]
if not dirs:
    raise RuntimeError("No output directories found.")
# Select the most recent one (by name, assuming timestamp format)
data_dir = sorted(dirs)[-1]
print(f"Using data directory: {data_dir}")

data = {}

# Load all .fits files in the directory
for fname in glob.glob(os.path.join(data_dir, "*.fits")):
    key = os.path.splitext(os.path.basename(fname))[0]
    with fits.open(fname) as hdul:
        arr = hdul[0].data
    data[key] = arr
    print('key:', key, 'type:', type(data[key]))

# Load all .pickle files in the directory
for fname in glob.glob(os.path.join(data_dir, "*.pickle")):
    key = os.path.splitext(os.path.basename(fname))[0]
    with open(fname, "rb") as f:
        data[key] = pickle.load(f)
    print('key:', key, 'type:', type(data[key]))

# Plot the sr.fits file if present (assumed to be a 1D vector)
if "sr" in data:
    sr = data["sr"]
    print(f"The average Strehl Ratio after 50 iterations is: {sr[50:].mean():.4f}")
    plt.figure()
    plt.plot(sr, marker='o')
    plt.title("Strehl Ratio (sr.fits)")
    plt.xlabel("Frame")
    plt.ylabel("SR")
    plt.yscale('log')
    plt.grid(True)
else:
    print("sr.fits file not found in the directory.")
    

if "modes" in data and "dm_commands" in data:
    res = data["modes"]
    comm = data["dm_commands"]
    init = 50 # leave 50 iterations for bootstrapping
    turb = res[init:-1, :].copy()
    turb[:, :comm.shape[1]] += comm[init+1:, :]
    x = xp.arange(turb.shape[1])+1

    # Plot RMS of residuals, commands and turbulence
    plt.figure(figsize=(12, 6))
    plt.plot(x,xp.sqrt(xp.mean(turb**2, axis=0)), label='Turbulence RMS', marker='o')
    plt.plot(x,xp.sqrt(xp.mean(res**2, axis=0)), label='Residuals RMS', marker='o')
    plt.plot(x[:comm.shape[1]],xp.sqrt(xp.mean(comm**2, axis=0)), label='Commands RMS', marker='o')
    plt.title("RMS of Turbulence, Residuals and Commands")
    plt.xlabel("Mode number")
    plt.ylabel("RMS")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)


if "res_ef" in data:
    lambdaInNm = 750
    ef = data["res_ef"]
    ef_amp = ef[-1,0,:,:]
    ef_phase = ef[-1,1,:,:]
    ef_rec = ef_amp * xp.exp(1j*ef_phase/lambdaInNm*(2*xp.pi),dtype=xp.complex128)
    oversampling = 12
    padding_len = xp.max(ef_rec.shape)*(oversampling-1)//2
    ef_rec = xp.pad(ef_rec, padding_len, mode='constant', constant_values=0.0)
    ff = xp.fft.fftshift(xp.fft.fft2(ef_rec))
    psf = xp.real(ff * xp.conj(ff))

    pupil_mask = xp.real(ef_rec) > 0
    avg_ef = xp.sum(ef_rec * pupil_mask) / xp.sum(pupil_mask)
    u_ef = ef_rec - avg_ef * pupil_mask
    u_ff = xp.fft.fftshift(xp.fft.fft2(u_ef))
    coro_psf = xp.real(u_ff * xp.conj(u_ff))

    coro_psf /= xp.max(psf)
    psf /= xp.max(psf)

    plt.figure()
    showZoomCenter(psf,1/oversampling, title='PSF',
                   xlabel=r'$\lambda/D$',ylabel=r'$\lambda/D$',zlabel='Contrast')
    plt.figure()
    showZoomCenter(coro_psf,1/oversampling, title='Coronographic PSF',
                   xlabel=r'$\lambda/D$',ylabel=r'$\lambda/D$',zlabel='Contrast')


plt.show()

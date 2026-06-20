
from astropy.io import fits
import numpy as np
import os.path as op
import datetime


pyr_masks = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupmask_shift.fits').astype(bool)
pupids = fits.getdata('/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/pup_ids.fits')
filepath=f'/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils/lbt_pupdata.fits'
npix = 120
pup_hdu = fits.open(filepath)
pup_ids = pup_hdu[1].data
fimg = np.zeros(npix**2)

def generate_rec(im,Nmodes:int,argos:bool,rMod=3.0): #,iir_path:str='/raid1/mmenessini/calibration/SOUL/KLv30dx/data/iir_rows.fits'
    rec_hdr = fits.getheader('/raid1/mmenessini/calibration/SOUL/KLv30dx/data/Rec_LUCI2_IIR_bin1_500modes.fits').copy()
    aux = np.zeros_like(im)
    IM = np.zeros_like(im)
    # Adjust slopes
    aux[:2512//2,:] = im[2512//2:,:]
    aux[2512//2:,:] = im[:2512//2,:]*-1
    # Adjust scaling
    aux *= 4e+9
    half_mask = pyr_masks[:60,:120].astype(bool)
    img = np.zeros(np.size(half_mask))
    for i in range(aux.shape[1]):
        np.put(fimg, pup_ids[:,0], aux[:2512//2,i])
        np.put(fimg, pup_ids[:,1], aux[2512//2:,i])
        f2d = fimg.reshape([npix,npix])
        img = f2d[:60,:]
        IM[pupids,i] = img.flatten()[half_mask.flatten()]
    IMinv = np.linalg.pinv(IM[:2512,:Nmodes])
    Rec = np.pad(IMinv,pad_width=((0,672-Nmodes),(0,2848-2512)),mode='constant',constant_values=0.0)
    # Add IIR rows
    Rec[661,:] = np.pad(IMinv[0,:],pad_width=(0,2848-2512),mode='constant',constant_values=0.0)   
    Rec[668,:] = np.pad(IMinv[1,:],pad_width=(0,2848-2512),mode='constant',constant_values=0.0)  
    Rec = Rec.astype('>f4')
    if argos:
        Rec /= 2
    # Modify header
    rec_hdr['BINNING'] = 1
    rec_hdr['IM_MODES'] = Nmodes
    rec_hdr['M2C'] = 'KL_v30'
    rec_hdr['ORIG_REC'] = 'synth_rec'
    rec_hdr['C_DIST_F'] = 'synth_pp'
    rec_hdr['M_DIST_F'] = 'synth_pp'
    rec_hdr['DATE'] = datetime.datetime.now().strftime("%Y-%m-%d")
    rec_hdr['tt.LAMBDA_D'] = rMod
    return Rec,IM,rec_hdr


if __name__ == "__main__":
    impath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/im'
    recpath = '/raid1/mmenessini/calibration/SOUL/KLv30dx/rec'
    rMod = 0.0
    seeing = 1.0
    types = ['DL','PCinf','PCperf']
    Nmodes = np.array([100,500,550,600]) #200,300,400,
    for tp in types:
        if tp == 'DL':
            tag = f'pyr{rMod:1.1f}_40x40_lbt_optsynim'
            rectag = f'Rec_mod{rMod:1.1f}_synthDL' #_LowAmp
        elif tp == 'PCinf':
            tag = f'pyr{rMod:1.1f}_s{seeing:1.1f}_synim_pcinf' #_LowAmp
            rectag = f'Rec_mod{rMod:1.1f}_synthPCinf_s{seeing:1.1f}' #_LowAmp
        elif tp == 'PCperf':
            tag = f'pyr{rMod:1.1f}_s{seeing:1.1f}_synim_pcperf' #_LowAmp
            rectag = f'Rec_mod{rMod:1.1f}_synthPCperf_s{seeing:1.1f}' #_LowAmp
        print(tag)
        IntMat = fits.getdata(op.join(impath,tag+'.fits'))
        for N in Nmodes:
            Rec,_,rec_hdr = generate_rec(IntMat,N,argos=True,rMod=rMod)
            # fits.writeto(op.join(recpath,imtag+f'_lbtlike_im.fits'),IM,overwrite=True)
            # print(f'Saved im as {imtag}_lbtlike_im.fits')
            fits.writeto(op.join(recpath,rectag+f'_{N}modes.fits'),Rec,header=rec_hdr,overwrite=True)
            print(f'Saved rec as {rectag}_{N}modes.fits')

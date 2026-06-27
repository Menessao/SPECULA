import numpy as np
from astropy.io import fits
import os.path as op
import pandas as pd

from specula.mmlib.compute_rec import rec_phot_cov, rec_ron_cov
from specula.mmlib.utils import get_pupil_mask, compute_modal_variance_von_karman, radial_order
from specula.mmlib.filter_optimizers import optimize_int_controller, optimize_higher_order_scao

root_dir = '/raid1/mmenessini/calibration/RISTRETTOunobs'

kl_inv = fits.getdata(op.join(root_dir,'ifunc','bmc2k_vlt_kl_inv.fits'))
modal_basis = np.linalg.pinv(kl_inv)
pup_mask = fits.getdata('/raid1/mmenessini/calibration/RISTRETTO/pupilstop/vlt_pupil_160pixels.fits')

def get_rec_covariance(rMod,n_subap,seeing,Nphot,RON,N,res:str='',Nho_multistage:int=None):
    pyr_mask = get_pupil_mask(npix=120,filepath=op.join(root_dir,'pupils',f'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}.fits'))
    frame = fits.getdata(op.join(root_dir,'frames',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N}modes_avg_frame.fits'))
    sn = fits.getdata(op.join(root_dir,'slopenulls',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N}modes_sn.fits'))
    rec = fits.getdata(op.join(root_dir,'rec',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_{N}modes_rec.fits'))
    ron_cov = rec_ron_cov(rec,frame,mask=pyr_mask)*RON/Nphot**2 
    shot_cov = rec_phot_cov(rec,frame,mask=pyr_mask,sn=sn)/Nphot
    cov = ron_cov + shot_cov
    if Nho_multistage is not None:
        slopes = fits.getdata(op.join(root_dir,'aliasing',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{Nho_multistage}modes_slopes.fits'))
        alias = np.std(rec @ slopes.T,axis=1)
        try:
            simpc = fits.getdata(op.join(root_dir,'im',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{Nho_multistage}modes{res}_simpc.fits'))        
            og = np.diag(rec @ simpc)
            ct = np.sqrt(np.sum((rec @ simpc)**2,axis=0)-og**2)
        except:
            # print(f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{Nho_multistage}modes{res}_simpc.fits not found, fallback to 48x48')
            rec = fits.getdata(op.join(root_dir,'rec',f'pyr{rMod:1.1f}_48x48_{Nho_multistage}modes_rec.fits'))
            simpc = fits.getdata(op.join(root_dir,'im',f'pyr{rMod:1.1f}_48x48_s{seeing:1.2f}_{Nho_multistage}modes{res}_simpc.fits'))        
            og = np.diag(rec @ simpc)
            ct = np.sqrt(np.sum((rec @ simpc)**2,axis=0)-og**2)
    else:
        alias = fits.getdata(op.join(root_dir,'aliasing',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N}modes_alias.fits'))
        simpc = fits.getdata(op.join(root_dir,'im',f'pyr{rMod:1.1f}_{n_subap:.0f}x{n_subap:.0f}_s{seeing:1.2f}_{N}modes{res}_simpc.fits'))
        og = np.diag(rec @ simpc)
        ct = np.sqrt(np.sum((rec @ simpc)**2,axis=0)-og**2)
    return cov,og,ct,alias

def get_vkp_for_seeing(seeing,L0=25,D=8.2):
    fname = op.join(root_dir,'data',f'VonKarmanVariance_s{seeing:1.2f}_diam{D:1.2f}m_LO{L0:1.0f}m')
    try:
        vkp = fits.getdata(fname)
    except FileNotFoundError:
        r0 = 0.98*500e-9/seeing*180/np.pi*3600
        vkp = compute_modal_variance_von_karman(r0=r0,L0=L0,D=D,modes=modal_basis.T,mask=(1-pup_mask).astype(bool))
        fits.writeto(fname,vkp)
    return vkp

def define_error_budget(rMod, seeing, VoverD, RON, delayInS, n_subaps, idx_modes, Nho_multistage:int, nModesPerSubaps,
                        res_str='', fluxes = np.logspace(6,8,5),freqs = np.linspace(100,5000,25)):
    
    # resfs = np.zeros([len(fluxes),len(freqs),len(n_subaps),len(idx_modes)])
    # resfs_ms = np.zeros([len(fluxes),len(freqs),len(n_subaps),len(idx_modes)])
    vkp = get_vkp_for_seeing(seeing)

    save_str = f'rMod{rMod:1.1f}_s{seeing:1.2f}_RON{RON:1.1f}_delay{delayInS*1e+6}us_VoverD{VoverD}Hz_CL{res_str}.csv'
    fname = op.join(root_dir,'eb_csv','standard_INT_'+save_str)
    fname_ms = op.join(root_dir,'eb_csv',f'multistage{Nho_multistage}_INT_'+save_str)

    columns =  ['Nsubaps', 'freqInHz', 'photPerMs', 'modeId', 'gain', 'filter', 'residual']

    if op.exists(fname):
        results_df = pd.read_csv(fname)
    else:
        results_df = pd.DataFrame(columns=columns)

    if op.exists(fname_ms):
        results_ms_df = pd.read_csv(fname_ms)
    else:
        results_ms_df = pd.DataFrame(columns=columns)

    results = results_df.to_dict(orient='records')
    results_ms = results_ms_df.to_dict(orient='records')
    existing_keys = {
        (float(row['Nsubaps']), float(row['freqInHz']), float(row['photPerMs']), float(row['modeId']), row['filter'])
        for row in results
    }
    existing_keys_ms = {
        (float(row['Nsubaps']), float(row['freqInHz']), float(row['photPerMs']), float(row['modeId']), row['filter'])
        for row in results_ms
    }

    for l,idx in enumerate(idx_modes):
        print(f'Working on mode {idx:1.0f}')
        for i,flux in enumerate(fluxes):
            for j,fs in enumerate(freqs):
                Nphot = flux/fs
                delay = delayInS*fs+1.0
                for k,n_subap in enumerate(n_subaps):
                    Nm = nModesPerSubaps[k]
                    if idx < Nm:
                        key = (float(n_subap), float(fs), float(flux/1e+3), float(idx), 'INT')
                        # Standard AO
                        if key not in existing_keys:
                            cov,og,_,alias = get_rec_covariance(rMod=rMod,n_subap=n_subap,N=Nm,seeing=seeing,Nphot=Nphot,RON=RON,res=res_str,Nho_multistage=None)
                            res,g = optimize_int_controller(f_s=fs,turb_power=vkp[idx],noise_power=cov[idx]+alias[idx]**2,show=False,
                                                n=radial_order(idx),cutoff=0.3*(radial_order(idx)+1)*VoverD,delay_steps=delay,og=og[idx])
                            results.append({'Nsubaps': n_subap,'freqInHz': fs,'photPerMs': flux/1e+3, 'modeId': idx, 'filter': 'INT', 'gain': g, 'residual': res})
                            existing_keys.add(key)
                        # Multistage AO
                        if key not in existing_keys_ms:
                            cov,og,_,alias = get_rec_covariance(rMod=rMod,n_subap=n_subap,N=Nm,seeing=seeing,Nphot=Nphot,RON=RON,res=res_str,Nho_multistage=Nho_multistage)
                            res,g_ms = optimize_int_controller(f_s=fs,turb_power=vkp[idx],noise_power=cov[idx]+alias[idx]**2,show=False,
                                                n=radial_order(idx),cutoff=0.3*(radial_order(idx)+1)*VoverD,delay_steps=delay,og=og[idx])
                            results_ms.append({'Nsubaps': n_subap,'freqInHz': fs,'photPerMs': flux/1e+3, 'modeId': idx, 'filter': 'INT', 'gain': g_ms, 'residual': res})
                            existing_keys_ms.add(key)
                    
    results_df = pd.DataFrame(results, columns=columns) 
    results_df.to_csv(fname, index=False)

    results_ms_df = pd.DataFrame(results_ms, columns=columns) 
    results_ms_df.to_csv(fname_ms, index=False)

                

if __name__ == '__main__':
    
    rMods = np.array([0,1,2,3])
    seeings = np.array([0.9,0.5,0.7,1.1,1.3,1.5])

    delayInS = 500e-6 # delay
    V = 12
    rMod = 0
    RON = 0.5
    res_str = '_1Nm'

    D = 8.2
    VoverD = V/D
    Nmodes = 1200
    idx_modes = np.arange(150)
    n_subaps = np.array([12,16,24,48])
    nModesPerSubaps = np.array([50,150,300,1200])

    for seeing in seeings:
        for rMod in rMods:
            print(f'Testing seeing {seeing:1.2f}", rMod = {rMod:1.1f}')
            define_error_budget(rMod, seeing, VoverD=VoverD, RON=RON, delayInS=delayInS, n_subaps=n_subaps, nModesPerSubaps=nModesPerSubaps, idx_modes=idx_modes, Nho_multistage=Nmodes, res_str=res_str)


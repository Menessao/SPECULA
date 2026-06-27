import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def gen_synth_psd(f, cutoff, n, turb_power, noise_power):
    f_safe = np.maximum(f, f[0])
    if n == 1:
        psd_turb = np.where(f_safe <= cutoff, (f_safe / cutoff)**(-2/3), (f_safe / cutoff)**(-17/3))
    else:
        psd_turb = np.where(f_safe <= cutoff, 1.0, (f_safe / cutoff)**(-17/3))
        
    df = np.gradient(f)
    f_s = f[-1] * 2 
    sinc_filter = np.sinc(f_safe / f_s)**2 
    psd_turb *= sinc_filter
    
    psd_turb *= (turb_power / np.sum(psd_turb * df))
    
    psd_noise = np.ones_like(f)
    psd_noise *= (noise_power / np.sum(psd_noise * df))
    
    return psd_turb, psd_noise

def eval_tf(b, a, f, f_s):
    z_inv = np.exp(-1j * 2 * np.pi * f / f_s)
    num = sum(b[k] * (z_inv**k) for k in range(len(b)))
    den = sum(a[k] * (z_inv**k) for k in range(len(a)))
    return num / den

def is_stable(a):
    roots = np.roots(a)
    return np.all(np.abs(roots) < 0.98)

def pre_fit_controller(order, f, psd_turb, f_s):
    target_mag = np.sqrt(psd_turb)
    target_mag /= np.max(target_mag)
    
    def loss(params):
        b = params[:order+1]
        a = np.insert(params[order+1:], 0, 1.0)
        if not is_stable(a): return 1e6
        C = eval_tf(b, a, f, f_s)
        return np.sum((np.abs(C) - target_mag)**2)

    init_b = np.zeros(order + 1); init_b[0] = 0.1
    init_a = np.zeros(order); init_a[0] = -0.9
    init_params = np.concatenate([init_b, init_a])

    res = minimize(loss, init_params, method='Nelder-Mead', options={'maxiter': 5000})
    b_guess = res.x[:order+1]
    a_guess = np.insert(res.x[order+1:], 0, 1.0)
    return b_guess, a_guess

def cost_function(params, order, f, psd_turb, psd_noise, N, f_s, og=1.0, lambda_reg=1e-3):
    """
    Cost function with L2 Regularization on the numerator coefficients (b)
    to smoothly penalize excessively large gains.
    """
    if order == 0:
        # Simple integrator
        b = np.array([params[0]])
        a = np.array([1.0, -1.0])
    else:
        b = params[:order+1]
        a = np.insert(params[order+1:], 0, 1.0)
        if not is_stable(a): return 1e9 

    C = eval_tf(b, a, f, f_s)
    P = np.exp(-1j * 2 * np.pi * f * N / f_s)
    H_ol = C * P
    denom = 1.0 + H_ol*og
    
    # # Avoid div by zero if perfectly touching the -1 point
    # if np.min(np.abs(denom)) < 1e-4: return 1e9

    RTF = 1.0 / denom
    NTF = H_ol / denom
    psd_res = (np.abs(RTF)**2 * psd_turb) + (np.abs(NTF)**2 * psd_noise)
    
    # Base variance cost
    variance_cost = np.sum(psd_res * np.gradient(f))
    
    # # Smooth L2 Penalty on the numerator magnitude
    # # This prevents the coefficients from exploding to artificially force a fit
    if order == 0:
        l2_penalty = lambda_reg * np.sum(b**2) + np.sum(b**4)*(b>=1.9)
    else:
        l2_penalty = lambda_reg * np.sum(b**2)
    
    return variance_cost + l2_penalty



def optimize_higher_order_scao(f_s, orders=[0, 1, 2, 3], turb_power=100.0, noise_power=0.5, og=1.0, show:bool=False,
                               delay_steps=2.0, cutoff=15.0, n=1, restarts:int=100, alias_psd=0.0,
                               lambda_reg=1e-2):
    f = np.linspace(0.1, f_s/2.0, 2000)
    psd_turb, psd_noise = gen_synth_psd(f, cutoff, n, turb_power, noise_power)

    rms_ratio = []
    a_coeffs = []
    b_coeffs = []
    
    if show:
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.loglog(f, psd_turb/psd_noise, label=f'Turbulence PSD (n={n})', c='k')
        plt.loglog(f, psd_noise/psd_noise, ':',label='Noise Floor', c='gray')
        plt.subplot(1, 2, 2)
        plt.loglog(f, psd_turb, label='Uncorrected Turb', c='k')
        plt.loglog(f, psd_noise+alias_psd, ':', label='Noise floor', c='gray')
    

    for i, order in enumerate(orders):
        if order == 0:
            init_params = np.array([0.5])
        else:
            b_guess, a_guess = pre_fit_controller(order, f, psd_turb, f_s)
            init_params = np.concatenate([b_guess, a_guess[1:]])
        
        res = minimize(cost_function, init_params, 
                       args=(order, f, psd_turb, psd_noise+alias_psd, delay_steps, f_s, og, lambda_reg),
                       method='Nelder-Mead', 
                       options={'maxiter': 5000, 'adaptive': True})
        
        if order == 0:
            best_b = np.array([res.x[0]]) if isinstance(res.x, (list, np.ndarray)) else np.array([res.x])
            best_a = np.array([1.0, -1.0])
        else:
            best_b = res.x[:order+1]
            best_a = np.insert(res.x[order+1:], 0, 1.0)
        
        if max(abs(best_b)) < 5e-2: # do not control if gains are too low
            best_b *= 0.0

        C = eval_tf(best_b, best_a, f, f_s)
        P = np.exp(-1j * 2 * np.pi * f * delay_steps / f_s)
        H_ol = C * P
        den = 1.0 + H_ol*og
        RTF = 1.0 / den
        NTF = H_ol / den
        
        # Calculate actual variance without the artificial L2 penalty for accurate reporting
        psd_res = (np.abs(RTF)**2 * psd_turb) + (np.abs(NTF)**2 * (psd_noise+alias_psd))
        min_variance = np.sum(psd_res * np.gradient(f))
        turb_variance = np.sum(psd_turb * np.gradient(f))

        rms_ratio.append(np.sqrt(min_variance))#/turb_variance))
        if order > 0:
            a_coeffs.append(best_a)
            b_coeffs.append(best_b)
        
        if show:
            name = "Integrator" if order == 0 else f"Order {order}"
            print(f"{name}:")
            print(f" B coefficients: {best_b}")
            print(f" A coefficients: {best_a}")
            print(f" Residual Variance: {min_variance:.4e}\n")
            
            plt.subplot(1, 2, 1)
            plt.loglog(f, 1.0/(np.abs(RTF)**2), label=r'$|RTF|^{-2}$'+f' ({name})')
            plt.subplot(1, 2, 2)
            plt.loglog(f, psd_res, label=f'Residual ({name})')

    if show:
        plt.subplot(1, 2, 1)
        plt.title(f"Spectral SNR")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power / Amplitude")
        plt.grid(True, which="both", alpha=0.5)
        plt.legend()
        plt.xlim(f[0], f[-1])
        plt.ylim([1e-1,psd_turb[0]/psd_noise[0]*4])
        
        plt.subplot(1, 2, 2)
        plt.title(f"Residual PSD")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density")
        plt.grid(True, which="both", alpha=0.5)
        plt.legend()
        plt.xlim(f[0], f[-1])
        plt.ylim([1e-1*psd_noise[0],psd_turb[0]*4])

        plt.tight_layout()

    return np.array(rms_ratio), a_coeffs, b_coeffs



def int_cost_function(gain,f,f_s,delay_steps,psd_turb,psd_noise,og=1.0):
    den = np.array([1.0,-1.0])
    C = eval_tf(np.array([gain]), den, f, f_s)
    P = np.exp(-1j * 2 * np.pi * f * delay_steps / f_s)
    H_ol = C * P
    den = 1.0 + H_ol*og
    RTF = 1.0 / den
    NTF = H_ol / den
    psd_res = (np.abs(RTF)**2 * psd_turb) + (np.abs(NTF)**2 * psd_noise)
    variance = np.sum(psd_res * np.gradient(f))
    g_crit = 2*np.sin(np.pi/(4*delay_steps-2))
    penalty = variance*gain**2 * (gain>=g_crit*0.98)
    return variance + penalty

def optimize_int_controller(f_s, turb_power=100.0, noise_power=0.5, og=1.0, show:bool=False,
                               delay_steps=2.0, cutoff=15.0, n=1, alias_psd=0.0):

    f = np.linspace(0.1, f_s/2.0, 2000)
    psd_turb, psd_noise = gen_synth_psd(f, cutoff, n, turb_power, noise_power)
    
    if show:
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.loglog(f, psd_turb/psd_noise, label=f'Turbulence PSD (n={n})', c='k')
        plt.loglog(f, psd_noise/psd_noise, ':',label='Noise Floor', c='gray')
        plt.subplot(1, 2, 2)
        plt.loglog(f, psd_turb, label='Uncorrected Turb', c='k')
        plt.loglog(f, psd_noise+alias_psd, ':', label='Noise floor', c='gray')
    

    init_params = np.array([0.5])
    res = minimize(int_cost_function, init_params, 
                    args=(f, f_s, delay_steps, psd_turb, psd_noise+alias_psd, og),
                    method='Nelder-Mead', 
                    options={'maxiter': 5000, 'adaptive': True})
    
    best_gain = res.x[0]
    den = np.array([1.0,-1.0])
    C = eval_tf(np.array([best_gain]), den, f, f_s)
    P = np.exp(-1j * 2 * np.pi * f * delay_steps / f_s)
    H_ol = C * P
    den = 1.0 + H_ol*og
    RTF = 1.0 / den
    NTF = H_ol / den
    psd_res = (np.abs(RTF)**2 * psd_turb) + (np.abs(NTF)**2 * psd_noise)
    variance = np.sum(psd_res * np.gradient(f))

    return np.sqrt(variance), best_gain
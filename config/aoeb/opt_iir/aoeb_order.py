"""
Residual per mode vs radial order (as iir_study.py `order`) with the inputs of the aoeb error budget on the e2e (WFSW) products.

  cd SPECULA/config; python aoeb/opt_iir/aoeb_order.py        # one figure per total delay, next to this file

Sensor: pyramid rMod 1, 48x48, 900 nm, 1300 modes, seeing 0.9", fs 2 kHz (fixed), magnitudes at 900 nm.
From the aoeb machine (aoeb/e2e calibration.machine(), budget_wfsw.yml) for each radial order n, as the mean over the modes of n:
  turbulence variance (KL von Karman, r0 of the seeing), temporal PSD (machine's model: Conan, effective wind of ristretto_main),
  noise (photon + readout, measured slope-noise propagation, detector of budget_wfsw.yml) / og^2,
  aliasing (measured modal std, perfect correction of 1300 modes) / og^2 with the machine's aliasing spectrum.
Designs in nm^2 (OPD); the figures show the rms per mode [nm]. One representative mode per order (the order means) is designed with every controller, as
iir_study.design: integrator, leaky integrator, IIR Wiener N = 2, 3 and the Wiener bound; optical gain known (POL exact), no
modulus margin (as the error-budget notebook, max_sensitivity = None). Total delay = 1 + tau frames (tau = aoeb latency in frames).
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, CONFIG, os.path.join(CONFIG, 'aoeb', 'e2e')]

RMOD, NSUB, WL, SEEING, N, FS = 1.0, 48, 900.0, 0.9, 1300, 2000.0
MAGS = (5.0, 8.0, 11.0)                 # at 900 nm
DELAYS = (1.5, 2.0, 2.5)                # total delay [frames]
RADIAL = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25, 30]
ORDERS = (2, 3)                         # IIR orders (bound always added)
CACHE = os.path.join(HERE, 'aoeb_order_cache.npz')


def inputs():
    """Per radial order: dict of mean turbulence / noise / aliasing [nm^2] (noise per magnitude), og, wind, alias spectrum."""
    import yaml, importlib.util
    from specula.lib.n_phot import n_phot
    from aoeb import utils
    spec = importlib.util.spec_from_file_location('aoeb_e2e_utils', os.path.join(CONFIG, 'aoeb', 'e2e', 'utils.py'))
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    m = e2e.machine()
    Y = yaml.safe_load(open(os.path.join(CONFIG, 'aoeb', 'e2e', 'ristretto_main.yml')))
    area = (m.pupil() > 0).sum() * Y['main']['pixel_pitch'] ** 2
    cal = m.calibration(RMOD, NSUB, SEEING, N, N, ('svd',), WL)
    turb, rms, _ = m.turbulence(SEEING, N)
    alias = cal.alias_rms ** 2 * (SEEING / cal.alias_seeing) ** (5 / 3)
    o = utils.radial_order(np.arange(N))
    noise = {}
    for mg in MAGS:
        flux = n_phot(mg, lambda_=WL * 1e-9, width=1e-9)[0] * Y['cred_vis']['bandw'] * area      # [ph/s], as Study.flux_of
        ph, ro = cal.noise.modal_variance_parts(m.detector.n_det(flux, FS), m.detector)
        noise[mg] = (ph + ro) / cal.og ** 2 * rms ** 2
    mean = lambda v: {n: float(np.mean(v[o == n])) for n in RADIAL}
    return dict(turb=mean(turb * rms ** 2), alias=mean(alias / cal.og ** 2 * rms ** 2), og=mean(cal.og),
                noise={mg: mean(v) for mg, v in noise.items()}, wind=m.temporal or m.wind, D=m.D, alias_spec=m.alias_spectrum(WL))


def task(args):
    """Residuals [int, leaky, IIR N..., bound] of one (delay, magnitude, radial order) [nm^2]."""
    total, n, turb, noise, alias, wind, D, alias_spec = args
    from aoeb import filter_optimizers as fo
    from wiener_ao_iir import ModeSpectra, wiener_iir
    from ao_tools import closed_loop_residual, optimal_integrator, optimal_leaky_integrator
    spec = ModeSpectra(lambda f: turb * fo.unit_psd(f, n, wind, D), FS, total - 1.0,
                       lambda f: alias * alias_spec.unit_psd(f), noise=noise, M=2 ** 16 if n < 4 else 2 ** 15)
    _, ji = optimal_integrator(spec)
    _, jl = optimal_leaky_integrator(spec)
    info = wiener_iir(spec, max(ORDERS))[2]
    out = [ji, jl] + [info['lower'][k - 1][2] for k in ORDERS] + [info['bound']]
    print(f'delay {total} n {n} noise {noise:.3g}: ' + ' '.join(f'{x:.3g}' for x in out), flush=True)
    return out


def compute():
    """res[delay, mag, order, controller] [nm^2], and the inputs."""
    I = inputs()
    shape = (len(DELAYS), len(MAGS), len(RADIAL), len(ORDERS) + 3)
    key = repr((RMOD, NSUB, WL, SEEING, N, FS, MAGS, DELAYS, RADIAL, ORDERS))
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if str(z['key']) == key:
            return z['res'], I
    jobs = [(t, n, I['turb'][n], I['noise'][mg][n], I['alias'][n], I['wind'], I['D'], I['alias_spec'])
            for t in DELAYS for mg in MAGS for n in RADIAL]
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    import multiprocessing as mp
    with mp.get_context('fork').Pool(min(len(jobs), int(os.environ.get('AOEB_WIENER_JOBS', 32)))) as pool:
        res = np.array(pool.map(task, jobs, chunksize=1)).reshape(shape)
    np.savez(CACHE, res=res, key=key)
    return res, I


def plot(res, I):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from iir_study import styles
    labels, colors, markers = styles(ORDERS)
    ls = ['-', '--', ':']
    out = []
    for i, total in enumerate(DELAYS):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for k, mg in enumerate(MAGS):
            ax.plot(RADIAL, [np.sqrt(I['turb'][n]) for n in RADIAL], ls[k], color='0.75', lw=1) if k == 0 else None
            for c in range(res.shape[-1]):
                ax.plot(RADIAL, np.sqrt(res[i, k, :, c]), ls[k], color=colors[c], marker=markers[c], ms=4, lw=1.3, mfc='none')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(RADIAL); ax.set_xticklabels(RADIAL); ax.minorticks_off()
        ax.set(xlabel='Radial order n', ylabel='Residual per mode [nm rms]', title=f'Total delay {total:g} frames')
        ax.grid(True, which='both', alpha=0.3)
        h = [Line2D([], [], color=c, lw=1.5, marker=mk, mfc='none') for c, mk in zip(colors, markers)]
        h += [Line2D([], [], color='gray', ls=s) for s in ls[:len(MAGS)]] + [Line2D([], [], color='0.75')]
        lab = labels + [f'mag {mg:g} (900 nm)' for mg in MAGS] + ['open loop (turbulence)']
        ax.legend(h, lab, fontsize=8, ncol=2)
        ogs = [I['og'][n] for n in RADIAL]
        fig.suptitle(f'Pyramid rMod {RMOD:g}, {NSUB}x{NSUB}, {WL:.0f} nm, {N} modes, seeing {SEEING:g}", fs = {FS:g} Hz, aoeb e2e '
                     f'products (Conan, wind {I["wind"]:.1f} m/s; measured noise, aliasing, og {min(ogs):.2f}-{max(ogs):.2f}); '
                     'mean mode of each radial order', fontsize=8, wrap=True)
        fig.tight_layout()
        fname = os.path.join(HERE, f'residual_vs_radial_order_aoeb_delay{total:g}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        out.append(fname)
    return out


if __name__ == '__main__':
    res, I = compute()
    for f in plot(res, I):
        print(f)

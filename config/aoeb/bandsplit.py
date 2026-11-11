"""Band split of the 700-1500 nm light between two fused pyramids, chromatic VIS case (science 550 nm), on the WFSW error budget
(2026-11-10, config/chromatic_band_split.ipynb).

  Photometry     a sensor of band [lo, hi] nm collects the star spectrum integrated over the band (flat QE; magnitude linear in the
                 wavelength through m(900) and m(1300) = m(900) - colour, extrapolated over 700-1500 nm). Its products are those of the
                 photon-weighted mean wavelength (machine.source_wavelength / A37), its refractivity the photon-weighted mean n - 1.
  Fusion         budget_fused of the two sensors (same loop rate), three weightings of every mode's two measurements:
                   'min_noise'  inverse-variance (e2e Kalman filter, budget_fused default);
                   'uniform'    min_noise + one shift d for all modes, chosen on the contrast-zone proxy (Study.zone_psd);
                   'per_mode'   per mode, the weight minimising loop noise + aliasing + chromatic shear of that mode, closed form on the
                                transfer factors of the min_noise loop (one loop re-solve): the two bands are EXTRAPOLATED towards
                                550 nm where the chromatic shear wins over the noise amplification (a two-colour estimator).
  Chromatic      per mode: effective refractivity dn_j = sum_i w_ij (n_i - n_550) of the fused measurement; the shear PSD uses dn at
                 every spatial frequency (mode j at k_j = (n_j + 1) / 2D, radial order n_j), exact J0 form of chromatic.py (A24).
                 The single-beam model of budget_fused is replaced by it in result.chrom_bands / extra['chrom_shear'].
"""
import dataclasses

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j0

from . import utils

_trapz = getattr(np, 'trapezoid', None) or np.trapz
WL_RANGE = (700.0, 1500.0)


class Photometry:
    """Band photometry and refractivity of the star (colour = m900 - m1300) for a pupil `area` [m^2]."""

    def __init__(self, colour, area, chromatic, step=5.0, wl_range=WL_RANGE):
        from specula.lib.n_phot import n_phot
        self.colour, self.area, self.step = float(colour), float(area), float(step)
        self.lam = np.arange(wl_range[0] + step / 2, wl_range[1], step)
        self.ph0 = np.array([n_phot(0.0, lambda_=l * 1e-9, width=1e-9)[0] for l in self.lam])     # ph/s/m^2/nm at magnitude 0
        self.n1 = np.array([float(chromatic.n_minus_1(l)) for l in self.lam])

    def band(self, mag900, lo, hi):
        """dict(flux [ph/s], wavelength_nm (photon-weighted mean), n_minus_1 (photon-weighted mean), band) of the band [lo, hi] nm."""
        sel = (self.lam > lo) & (self.lam < hi)
        mag = mag900 - self.colour * (self.lam[sel] - 900.0) / 400.0
        ph = self.ph0[sel] * 10 ** (-0.4 * mag) * self.step * self.area
        return dict(flux=float(ph.sum()), wavelength_nm=float(np.sum(ph * self.lam[sel]) / ph.sum()),
                    n_minus_1=float(np.sum(ph * self.n1[sel]) / ph.sum()), band=(float(lo), float(hi)))

    def sensor(self, rMod, n_subap, mag900, lo, hi):
        return dict(rMod=float(rMod), n_subap=int(n_subap), **self.band(mag900, lo, hi))


@dataclasses.dataclass
class PerModeShear:
    """Chromatic shear PSD with an effective refractivity difference dn(k) (from per-mode weights): duck-types ChromaticModel.psds for
    result.chrom_bands (PsfEngine.psd_terms)."""
    model: object                 # ChromaticModel (zenith, layers, science wavelength)
    k_modes: np.ndarray           # spatial frequency of every mode [c/m]
    dn_modes: np.ndarray          # effective n_sensing - n_science of every mode

    def psds(self, k, r0, L0, k_cut, sensing_nm=None, k_max=None):
        k = np.asarray(k, float)
        order = np.argsort(self.k_modes)
        dn = np.interp(k, self.k_modes[order], self.dn_modes[order])
        a = self.model.slant_heights() * np.tan(np.radians(self.model.zenith_deg))          # shift per unit dn [m]
        w = np.asarray(self.model.cn2, float)[:, None]
        phi = utils.von_karman_psd(k, r0, L0)
        return {'chrom_shear': phi * (w * 2.0 * (1.0 - j0(2 * np.pi * k[None] * a[:, None] * dn[None]))).sum(0) * (k < k_cut)}


def mode_frequencies(m, N):
    """Spatial frequency [c/m] of the first N KL modes: (radial order + 1) / 2D."""
    return (utils.radial_order(np.arange(N)) + 1.0) / (2.0 * m.D)


def _sensor_terms(m, s, seeing, fs, N, rec=('svd',)):
    """(noise variance per frame [command^2], aliasing rms [command]) of every mode of one sensor, as budget_fused computes them."""
    n_rec = int(s.get('N_rec') or N)
    cal = m.calibration(s['rMod'], s['n_subap'], seeing, n_rec, int(s.get('N_corr') or n_rec), rec, s['wavelength_nm'])
    g = np.maximum(cal.og[:N], 1e-3)
    p, rn = cal.noise.modal_variance_parts(m.detector.n_det(s['flux'], fs), m.detector)
    return (p[:N] + rn[:N]) / g ** 2, cal.alias_rms[:N] * (seeing / cal.alias_seeing) ** (5 / 6) / g


class SplitStudy:
    """Fused budgets of band-split sensors with the chromatic shear per mode, on a multiwfs.Study (its machine, PSF engine, seeing,
    controller, latency, contrast zone and disk cache)."""

    def __init__(self, st, chromatic):
        self.st, self.m, self.cm = st, st.m, dataclasses.replace(chromatic, multiplicative=False, fresnel=False, scintillation=False)
        self.n_sci = float(self.cm.n_minus_1(self.cm.science_nm))
        st.set_band(self.cm.science_nm, self.cm)
        self._rates = {}

    def _dn(self, s):
        return float(s.get('n_minus_1') or self.cm.n_minus_1(s['wavelength_nm'])) - self.n_sci

    def _shear_per_dn2(self, N, seeing):
        """Per-mode chromatic shear variance [nm^2] per unit dn^2 in the small-shift limit (2 (1 - J0(x)) ~ x^2 / 2)."""
        turb, rms, r0 = self.m.turbulence(seeing, N)
        k = mode_frequencies(self.m, N)
        a = self.cm.slant_heights() * np.tan(np.radians(self.cm.zenith_deg))
        return turb * rms ** 2 * np.sum(np.asarray(self.cm.cn2, float)[:, None] * (2 * np.pi * k[None] * a[:, None]) ** 2 / 2, axis=0)

    def finalize(self, r, sensors, chromatic=True):
        """Per-mode chromatic shear of a fused result (weights in r.weights), in place: chrom_bands, extra['chrom_shear'], params."""
        N = len(r.gain)
        if not chromatic:
            r.chrom_bands, r.extra = None, {k: v for k, v in r.extra.items() if not k.startswith('chrom')}
            return r
        dn = np.sum(np.asarray(r.weights)[:, :N] * np.array([self._dn(s) for s in sensors])[:, None], axis=0)
        k = mode_frequencies(self.m, N)
        k_cut = utils.fitting_cutoff(N, self.m.D)
        shear = PerModeShear(self.cm, k, dn)
        r0 = float(utils.r0_from_seeing(self.st.seeing))
        kk = np.linspace(0.0, k_cut, 20000)
        r.extra = {**{x: v for x, v in r.extra.items() if not x.startswith('chrom')},
                   'chrom_shear': float(_trapz(shear.psds(kk, r0, self.m.L0, k_cut)['chrom_shear'] * 2 * np.pi * kk, kk))}
        r.chrom_bands = [(shear, 0.0, k_cut, r.params['wavelength_nm'])]
        r.params.update(dn_eff_zone=float(np.average(dn, weights=self.st.w_zone[:N])) / 1e-6)    # zone-weighted dn [1e-6]
        return r

    def zone(self, r, chrom_var=None):
        """Contrast-zone proxy: mean residual PSD [nm^2 m^2] in the zone of the per-mode residual (+ chromatic shear variances)."""
        v = r.residual + (0.0 if chrom_var is None else chrom_var)
        return self.st.zone_psd(v)

    def fused(self, sensors, weights='min_noise', chromatic=True, fs=None):
        """Fused budget of `sensors` (one or two) with the given weighting; fs None: rate of the grid minimising the total residual of the
        min_noise system (the weighting is then optimised at that rate). Returns the BudgetResult (params['weights'], ['weight_shift_zone']).
        Study criterion 'contrast' and fs None (multiwfs): one sensor -> the (binning, rate) of st.binnings x the rate grid ranked first
        by st._contrast_key (each binning with all its calibrated modes); two sensors -> the rate ranking the min_noise system first,
        then the weighting at that rate."""
        m, st = self.m, self.st
        if fs is None and getattr(st, 'criterion', 'strehl') == 'contrast':
            if len(sensors) == 1:
                cands = [self.fused([{**sensors[0], 'n_subap': n}], weights, chromatic, fs=f)
                         for n in st.binnings for f in st.fs_grid]
                return min(cands, key=st._contrast_key)
            key = (repr(sensors), bool(chromatic), st.sci_nm)              # the rate of a pair is shared by its three weightings
            if key not in self._rates:
                self._rates[key] = min(st.fs_grid, key=lambda f: st._contrast_key(self.fused(sensors, 'min_noise', chromatic, fs=f)))
            return self.fused(sensors, weights, chromatic, fs=self._rates[key])
        kw = {**st.kw, 'optimize_freq': fs is None}
        cm = self.cm if chromatic else None
        base = m.budget_fused(sensors, st.seeing, fs or st.fs_max, chromatic=cm, warn=False, **kw)
        N, f = len(base.gain), base.params['freqUsedInHz']
        if weights == 'min_noise' or len(sensors) == 1 or not chromatic:
            base.params['weights'] = 'min_noise'
            return self.finalize(base, sensors, chromatic)
        S = self._shear_per_dn2(N, st.seeing)
        dn = np.array([self._dn(s) for s in sensors])
        w0 = np.asarray(base.weights)[0, :N]
        run = lambda d: m.budget_fused(sensors, st.seeing, f, N=N, chromatic=cm, warn=False, weight_shift=d, **{**st.kw, 'optimize_freq': False})
        chrom_of = lambda w: S * (w * dn[0] + (1 - w) * dn[1]) ** 2
        if weights == 'uniform':
            cost = lambda d: self.zone(run(d), chrom_of(w0 + d))
            d = float(minimize_scalar(cost, bounds=(-4.0, 4.0), method='bounded', options=dict(xatol=0.01)).x)
            r = run(d)
        elif weights == 'per_mode':
            (s0, a0), (s1, a1) = (_sensor_terms(m, s, st.seeing, f, N) for s in sensors)
            r = base
            for _ in range(2):                      # transfer factors of the current loop, closed-form weights, loop re-solved
                _, rms, _ = m.turbulence(st.seeing, N)
                w = np.asarray(r.weights)[0, :N]
                n_in = w ** 2 * s0 + (1 - w) ** 2 * s1
                a_in = (w * a0 + (1 - w) * a1) ** 2
                t_n = (r.noise / rms ** 2) / np.maximum(n_in, 1e-300)
                t_a = (r.alias / rms ** 2) / np.maximum(a_in, 1e-300)
                Sc = S / rms ** 2                                            # chromatic shear [command^2] per unit dn^2
                num = t_n * s1 - t_a * (a0 - a1) * a1 - Sc * (dn[0] - dn[1]) * dn[1]
                den = t_n * (s0 + s1) + t_a * (a0 - a1) ** 2 + Sc * (dn[0] - dn[1]) ** 2
                w_new = np.clip(num / np.maximum(den, 1e-300), -6.0, 7.0)
                r = run(w_new - w0)
            d = float(np.average(np.asarray(r.weights)[0, :N] - w0, weights=st.w_zone[:N]))
        else:
            raise ValueError(f'weights must be min_noise, uniform or per_mode, not {weights!r}')
        r.params.update(weights=weights, weight_shift_zone=d)
        return self.finalize(r, sensors, chromatic)

    def row(self, r, **extra):
        """Contrast (zone, science wavelength), Strehl, chromatic and total residual [nm rms], frame rate, zone-weighted dn, weights."""
        v = r.variances()
        return dict(extra, contrast=self.st.contrast(r), sr=r.strehl(self.cm.science_nm), total_nm=np.sqrt(v['total']),
                    chrom_nm=np.sqrt(v.get('chrom_shear', 0.0)), fs=r.params['freqUsedInHz'], n_subap=r.params.get('Nsubaps', np.nan), dn_eff_zone=r.params.get('dn_eff_zone', np.nan),
                    weight_shift_zone=r.params.get('weight_shift_zone', 0.0))

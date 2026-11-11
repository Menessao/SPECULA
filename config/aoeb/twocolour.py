"""Two-colour correction of the chromatic shear (2026-11-11): two pyramids on EITHER SIDE of the science wavelength (400-500 nm and
700-800 nm for 600 nm), their per-mode measurements combined with weights whose effective refractivity sum_i w_i n_i is that of the
science wavelength, so that the chromatic shear cancels to first order. On bandsplit.SplitStudy (same loop, weighting and rate rules).

  Photometry     FlatPhotometry: flat photon spectrum, the photons per nm at every wavelength equal those of the star at 900 nm (mag900):
                 a band [lo, hi] collects (hi - lo) times the 900 nm photons per nm; its products are those of the band centre, its
                 refractivity the band mean of n - 1.
  Chromatic      MultiBeamShear, EXACT for any number of beams: the science phase is estimated by sum_i w_i phi(x + s_i), s_i = a dn_i the
                 shift of the layer seen by sensor i (a = h tan z / cos z, dn_i = n_i - n_sci); azimuthal average of
                 |sum_i w_i exp(2 pi i k.s_i) - 1|^2 per layer:
                     1 - 2 sum_i w_i J0(2 pi k a dn_i) + sum_ij w_i w_j J0(2 pi k a (dn_i - dn_j)),
                 = 2 (1 - J0(2 pi k a dn)) for one beam (chromatic.py). The first-order model of SplitStudy (one effective dn per mode) is
                 zero when sum_i w_i dn_i = 0; this one keeps the second-order residual (dn_1 dn_2 (2 pi k a)^2 / 2 per layer to leading
                 order). The per-mode weights are interpolated over the spatial frequency (mean weight of each radial order).
  Band contrast  band_contrast: a result (loop, weights: corrected for 600 nm) evaluated at every science wavelength of a band, the other
                 terms unchanged in nm, the shear with dn_i(lambda) = n_i - n(lambda), the PSF and the 2-5 lambda/D zone at lambda.
"""
import copy
import dataclasses

import numpy as np
from scipy.special import j0

from . import utils
from .bandsplit import SplitStudy, mode_frequencies

_trapz = getattr(np, 'trapezoid', None) or np.trapz


class FlatPhotometry:
    """Flat photon spectrum normalised at 900 nm, for a pupil `area` [m^2]; chromatic: ChromaticModel (refractivity)."""

    def __init__(self, area, chromatic, step=1.0):
        from specula.lib.n_phot import n_phot
        self.area, self.cm, self.step = float(area), chromatic, float(step)
        self.ph0 = float(n_phot(0.0, lambda_=900e-9, width=1e-9)[0])        # ph/s/m^2/nm at magnitude 0, 900 nm

    def band(self, mag900, lo, hi):
        lam = np.arange(lo + self.step / 2, hi, self.step)
        return dict(flux=self.ph0 * 10 ** (-0.4 * mag900) * (hi - lo) * self.area, wavelength_nm=float(lam.mean()),
                    n_minus_1=float(np.mean([float(self.cm.n_minus_1(x)) for x in lam])), band=(float(lo), float(hi)))

    def sensor(self, rMod, n_subap, mag900, lo, hi):
        return dict(rMod=float(rMod), n_subap=int(n_subap), **self.band(mag900, lo, hi))


@dataclasses.dataclass
class MultiBeamShear:
    """Exact chromatic shear PSD of a weighted sum of beams; duck-types ChromaticModel.psds for result.chrom_bands (PsfEngine)."""
    model: object                 # ChromaticModel (zenith, layers)
    k_modes: np.ndarray           # spatial frequency of every mode [c/m]
    weights: np.ndarray           # (n_beams, N) per-mode weights
    dn: np.ndarray                # (n_beams,) n_i - n_science

    def __post_init__(self):
        self.k_modes, self.weights, self.dn = np.asarray(self.k_modes, float), np.atleast_2d(self.weights), np.atleast_1d(self.dn)
        self._k, inv = np.unique(self.k_modes, return_inverse=True)            # one weight per radial order (mean over its modes)
        self._w = np.stack([np.bincount(inv, w) / np.bincount(inv) for w in self.weights])

    def factor(self, k):
        """(n_layers, len(k)) azimuthally averaged |sum_i w_i exp(2 pi i k.s_i) - 1|^2."""
        k = np.asarray(k, float)
        w = np.stack([np.interp(k, self._k, wi) for wi in self._w])           # (n_beams, K)
        a = self.model.slant_heights() * np.tan(np.radians(self.model.zenith_deg))
        x = lambda d: j0(2 * np.pi * k[None] * a[:, None] * d)                  # (n_layers, K)
        out = np.ones((a.size, k.size))
        for i, di in enumerate(self.dn):
            out -= 2 * w[i] * x(di)
            for j, dj in enumerate(self.dn):
                out += w[i] * w[j] * x(di - dj)
        return out

    def psds(self, k, r0, L0, k_cut, sensing_nm=None, k_max=None):
        k = np.asarray(k, float)
        w = np.asarray(self.model.cn2, float)[:, None]
        return {'chrom_shear': utils.von_karman_psd(k, r0, L0) * (w * self.factor(k)).sum(0) * (k < k_cut)}


class TwoColourStudy(SplitStudy):
    """SplitStudy with the exact multi-beam shear (MultiBeamShear) and the band evaluation of a result."""

    def finalize(self, r, sensors, chromatic=True, cm=None):
        N = len(r.gain)
        if not chromatic:
            return super().finalize(r, sensors, False)
        cm = cm or self.cm
        n_sci = float(cm.n_minus_1(cm.science_nm))
        dn = np.array([float(s.get('n_minus_1') or cm.n_minus_1(s['wavelength_nm'])) - n_sci for s in sensors])
        W = np.asarray(r.weights)[:, :N]
        k_cut = utils.fitting_cutoff(N, self.m.D)
        shear = MultiBeamShear(cm, mode_frequencies(self.m, N), W, dn)
        r0 = float(utils.r0_from_seeing(self.st.seeing))
        kk = np.linspace(0.0, k_cut, 20000)
        r.extra = {**{x: v for x, v in r.extra.items() if not x.startswith('chrom')},
                   'chrom_shear': float(_trapz(shear.psds(kk, r0, self.m.L0, k_cut)['chrom_shear'] * 2 * np.pi * kk, kk))}
        r.chrom_bands = [(shear, 0.0, k_cut, r.params['wavelength_nm'])]
        r.params.update(dn_eff_zone=float(np.average(dn @ W, weights=self.st.w_zone[:N])) / 1e-6,
                        w0_zone=float(np.average(W[0], weights=self.st.w_zone[:N])))
        return r

    def band_contrast(self, r, sensors, lams=np.arange(500.0, 701.0, 20.0)):
        """{lambda: (contrast, SR)} of result r (its loop and weights) at the science wavelengths lams, plus 'band' (mean contrast,
        mean SR); the Study's science wavelength is restored."""
        sci0, out = self.st.sci_nm, {}
        try:
            for lam in lams:
                cm = dataclasses.replace(self.cm, science_nm=float(lam))
                rl = self.finalize(copy.deepcopy(r), sensors, True, cm)
                self.st.sci_nm = float(lam)
                out[float(lam)] = (self.st.contrast(rl), rl.strehl(float(lam)))
        finally:
            self.st.sci_nm = sci0
        c = np.array([v[0] for v in out.values()])
        out['band'] = (float(np.nanmean(c)), float(np.mean([v[1] for v in out.values()])))
        return out

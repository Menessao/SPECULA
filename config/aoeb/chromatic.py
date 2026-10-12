"""Chromatic and scintillation terms as radial OPD PSDs [nm^2 m^2]; definitions and assumptions: ASSUMPTIONS.md (A24-A27).

  chrom_shear    refractive-path shear (TipTop differentialRefractionPSD), always on, k < k_c
  chrom_mult     multiplicative term (TipTop chromatismPSD), optional, per mode and loop dependent (budget.py)
  chrom_fresnel  optional: chromatic OPD of the Fresnel propagation, Phi sum_l w_l [cos(pi lc h k^2) - cos(pi ls h k^2)]^2, k < k_c
  scintillation  optional: log-amplitude at the science wavelength, Phi sum_l w_l sin^2(pi lc h k^2), all k < k_max
Layer profile, zenith and flags come from config/defaults.yml (atmosphere, chromatic). n - 1 comes from TipTop's MatharAirRefraction (`p3.aoSystem.airRefraction` or `airRefraction`) or an `air_model`; no fallback.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import j0

from . import utils

_trapz = getattr(np, 'trapezoid', None) or np.trapz
_TIPTOP = None

def tiptop_air_model():
    global _TIPTOP
    if _TIPTOP is None:
        errors = []
        for mod in ('p3.aoSystem.airRefraction', 'airRefraction'):
            try:
                _TIPTOP = __import__(mod, fromlist=['MatharAirRefraction']).MatharAirRefraction()
                break
            except Exception as e:                                   # noqa
                errors.append(f'{mod}: {e}')
        else:
            raise ImportError(f"TipTop's airRefraction.MatharAirRefraction is required (tried {errors}); put it on the path or pass air_model=")
    return _TIPTOP


def refractivity(wavelength_nm, T_celsius=11.5, P_pa=74300.0, RH_pct=10.0, air_model=None):
    """n - 1 of air (checked: a model returning n instead would be off by a factor ~3500)."""
    m = air_model if air_model is not None else tiptop_air_model()
    n = m.get_refractive_index(np.asarray(wavelength_nm, float) * 1e-9, T_celsius, P_pa, RH_pct)
    if not np.all((1e-5 < np.asarray(n)) & (np.asarray(n) < 1e-3)):
        raise ValueError(f'the air model must return n - 1 (~2e-4 near sea level / 7e-5..3e-4 at altitude), got {n}')
    return n


@dataclass
class ChromaticModel:
    heights: np.ndarray                  # layer altitudes at zenith [m]
    cn2: np.ndarray
    science_nm: float
    zenith_deg: float = 0.0
    T_celsius: float = 11.5
    P_pa: float = 74300.0
    RH_pct: float = 10.0
    multiplicative: bool = False         # loop-dependent term; compensation = fraction of eta removed by the controller
    compensation: float = 0.0
    fresnel: bool = False
    scintillation: bool = False
    k_max: float = None                  # [c/m] upper limit of the scintillation; None: the Nyquist frequency of the pupil grid
    air_model: object = None
    delta_n: float = None                # override of n_sensing - n_science
    heights_are_slant: bool = False      # heights already include the airmass (TipTop's Hs)

    def __post_init__(self):
        self.heights = np.atleast_1d(np.asarray(self.heights, float))
        w = np.atleast_1d(np.asarray(self.cn2, float))
        self.cn2 = w / w.sum()

    def slant_heights(self):
        return self.heights * (1.0 if self.heights_are_slant else 1.0 / np.cos(np.radians(self.zenith_deg)))

    def n_minus_1(self, wavelength_nm):
        return refractivity(wavelength_nm, self.T_celsius, self.P_pa, self.RH_pct, self.air_model)

    def eta(self, sensing_nm):
        n_s, n_c = float(self.n_minus_1(sensing_nm)), float(self.n_minus_1(self.science_nm))
        return (n_c - n_s) / n_s * (1.0 - self.compensation)

    def layer_shifts(self, sensing_nm):
        dn = self.delta_n if self.delta_n is not None else float(self.n_minus_1(sensing_nm) - self.n_minus_1(self.science_nm))
        return self.slant_heights() * dn * np.tan(np.radians(self.zenith_deg))

    def psds(self, k, r0, L0, k_cut, sensing_nm, k_max=None):
        """{name: radial PSD on k} of the enabled terms (the multiplicative term is per mode, see budget.py)."""
        k = np.asarray(k, float)
        k_max = self.k_max or k_max or float(k.max())
        phi, hs, w, band = utils.von_karman_psd(k, r0, L0), self.slant_heights()[:, None], self.cn2[:, None], k < k_cut
        lc, ls, k2 = self.science_nm * 1e-9, sensing_nm * 1e-9, k[None] ** 2
        out = {'chrom_shear': phi * (w * 2.0 * (1.0 - j0(2 * np.pi * k[None] * self.layer_shifts(sensing_nm)[:, None]))).sum(0) * band}
        if self.fresnel:
            d = np.cos(np.pi * lc * hs * k2) - np.cos(np.pi * ls * hs * k2)
            out['chrom_fresnel'] = phi * (w * d ** 2).sum(0) * band
        if self.scintillation:
            out['scintillation'] = phi * (w * np.sin(np.pi * lc * hs * k2) ** 2).sum(0) * (k < k_max)
        return out

    def variances(self, r0, L0, k_cut, sensing_nm, k_max=None, n=60000):
        """{term: variance [nm^2]}; k_max: Nyquist frequency of the pupil grid (scintillation upper limit) unless self.k_max."""
        kmax = self.k_max or k_max or k_cut
        k = np.linspace(0.0, max(k_cut, kmax if self.scintillation else 0.0), n)
        return {name: float(_trapz(p * 2 * np.pi * k, k)) for name, p in self.psds(k, r0, L0, k_cut, sensing_nm, kmax).items()}

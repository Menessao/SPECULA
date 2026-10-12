"""File layout shared by the generator and the budget machine (one place, one scheme).

Tag resolution (values are ROUNDED to it, in the generator and in the machine alike): seeing 0.1 arcsec, rMod 0.1, wavelength 1 nm,
L0 1 m. The generator simulates the rounded value, so a product is what its name says; the machine rescales the aliasing to the
requested seeing (alias variance ~ seeing^(5/3)).

Shared by every sensing wavelength (no wavelength in the name):
  pupilstop/<pupil_tag>.fits   ifunc/<dm_tag>_kl_inv.fits   pupils/pyr_pupdata_{n}x{n}.fits
  data/kl_variance_{pupil}_{kl}_L0{L0}_D{D}_pad{pad}.fits   data/kl_psd_kernels_{pupil}_{kl}_D{D}_pad{pad}.fits

Per sensing wavelength: <base> = pyr{r}_{n}x{n}[_wvl{lambda_nm}]  (suffix only if a wavelength is given)
  im/       <base>_im_dl.fits                                 (diffraction-limited IM, all modes)
            <base>_s{seeing}_{N}modes_im_pc.fits              (IM, N modes perfectly corrected)
            <base>_s{seeing}_{N}modes_im_pc_c{max}-{min}.fits  (partial correction, correction vector max_corr-min_corr)
  data/   correction_vector_{N}modes_c{max}-{min}.fits           (BaseValue read by SPECULA; max = min = 1: perfect correction)
  rec/      <base>_{N}modes_rec.fits                          (SVD); ..._rec_{ml|map}_{nph}ph_s{seeing}.fits
  aliasing/ <base>_s{seeing}_{N}modes_slopes.fits             (n_frames x n_slopes, N modes corrected)
            <base>_s{seeing}_{N}modes_turbmodes.fits          (n_frames x N: simulated turbulence modes)
            <base>_s{seeing}_{Nrec}modes_corr{Nc}_alias.fits  (std of rec_Nrec @ slopes_Nc)
  frames/   <base>_s{seeing}_{N}modes_avgframe.fits
  slopenulls/ <base>_s{seeing}_{N}modes_sn.fits               (mean slopes)
  throughput/ <base>_s{seeing}_{N}modes_thrp.fits             (fraction of detector counts in the pupils)
"""
import glob
import re
import os.path as op
from dataclasses import dataclass, replace
from typing import Optional



def snap_seeing(seeing):
    return round(float(seeing), 1)


@dataclass(frozen=True)
class Paths:
    root: str
    pupilstop_tag: str = 'vlt_pupil_160pixels'
    dm_tag: str = 'bmc2k_vlt'                   # DM data objects: {dm_tag}_ifunc, {dm_tag}_m2c, {dm_tag}_kl_inv
    wavelength_nm: Optional[float] = None       # sensing wavelength; None = no wavelength tag

    def at(self, wavelength_nm):
        """The same layout for another sensing wavelength."""
        return replace(self, wavelength_nm=wavelength_nm)

    # ---- tags
    def base(self, rMod, n_subap):
        tag = f'pyr{rMod:.1f}_{n_subap:.0f}x{n_subap:.0f}'
        return tag if self.wavelength_nm is None else f'{tag}_wvl{self.wavelength_nm:.0f}'

    def sim(self, rMod, n_subap, seeing, N):
        return f'{self.base(rMod, n_subap)}_s{seeing:.1f}_{N:.0f}modes'

    def _f(self, folder, name):
        return op.join(self.root, folder, name + '.fits')

    # ---- static inputs
    def pupilstop(self):
        return self._f('pupilstop', self.pupilstop_tag)

    @property
    def ifunc_name(self):
        return f'{self.dm_tag}_ifunc'

    @property
    def m2c_name(self):
        return f'{self.dm_tag}_m2c'

    @property
    def kl_inv_name(self):
        return f'{self.dm_tag}_kl_inv'

    def kl_inv(self):
        return self._f('ifunc', self.kl_inv_name)

    def correction_vector(self, N, max_corr, min_corr):
        """BaseValue file read by SPECULA (`scale_random.constant_mul_data` = its stem); same name as save_correction_vector."""
        return self._f('data', f'correction_vector_{N:.0f}modes_c{max_corr:1.2f}-{min_corr:1.2f}')

    def pupdata(self, n_subap):
        return self._f('pupils', f'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}')

    # ---- interaction matrices and reconstructors
    def im_dl(self, rMod, n_subap):
        return self._f('im', self.base(rMod, n_subap) + '_im_dl')

    def im_pc(self, rMod, n_subap, seeing, N, level=None):
        """IM measured with the N modes corrected by a correction vector: level = None (perfect correction, vector of ones) or
        (max_corr, min_corr) of the partial correction (save_correction_vector)."""
        suffix = '_im_pc' if level is None else f'_im_pc_c{level[0]:1.2f}-{level[1]:1.2f}'
        return self._f('im', self.sim(rMod, n_subap, seeing, N) + suffix)

    def pc_levels(self, rMod, n_subap, seeing, N):
        """(max_corr, min_corr) of the partial-correction IMs on disk, sorted; (1.0, 1.0) stands for the perfect-correction IM."""
        pat = self._f('im', self.sim(rMod, n_subap, seeing, N) + '_im_pc_c*-*')
        levels = []
        for p in glob.glob(pat):
            m = re.search(r'_im_pc_c([0-9.]+)-([0-9.]+)\.fits$', p)
            if m:
                levels.append((float(m.group(1)), float(m.group(2))))
        if op.exists(self.im_pc(rMod, n_subap, seeing, N)):
            levels.append((1.0, 1.0))
        return sorted(set(levels))

    def rec(self, rMod, n_subap, N, method='svd', n_phot=None, seeing=None):
        """SVD: flux independent. ML / MAP: depend on the photons per frame and on the seeing (noise weights, prior)."""
        tag = f'{self.base(rMod, n_subap)}_{N:.0f}modes_rec'
        if method != 'svd':
            tag += f'_{method}_{n_phot:.4g}ph_s{seeing:.1f}'
        return self._f('rec', tag)

    # ---- aliasing run products
    def slopes(self, rMod, n_subap, seeing, N):
        return self._f('aliasing', self.sim(rMod, n_subap, seeing, N) + '_slopes')

    def turbmodes(self, rMod, n_subap, seeing, N):
        return self._f('aliasing', self.sim(rMod, n_subap, seeing, N) + '_turbmodes')

    def avgframe(self, rMod, n_subap, seeing, N):
        return self._f('frames', self.sim(rMod, n_subap, seeing, N) + '_avgframe')

    def sn(self, rMod, n_subap, seeing, N):
        return self._f('slopenulls', self.sim(rMod, n_subap, seeing, N) + '_sn')

    def thrp(self, rMod, n_subap, seeing, N):
        return self._f('throughput', self.sim(rMod, n_subap, seeing, N) + '_thrp')

    def alias(self, rMod, n_subap, seeing, N_rec, N_corr):
        return self._f('aliasing', f'{self.base(rMod, n_subap)}_s{seeing:.1f}_{N_rec:.0f}modes_corr{N_corr:.0f}_alias')

    # ---- cached analytical quantities
    def kl_variance(self, L0, D, pad):
        return self._f('data', f'kl_variance_{self.pupilstop_tag}_{self.dm_tag}_L0{L0:.0f}_D{D:.2f}_pad{pad}')

    def kl_psd_kernels(self, D, pad):
        """Radial kernels |FT(P m_i)|^2 / A of every KL mode (psf.py), shared by all wavelengths."""
        return self._f('data', f'kl_psd_kernels_{self.pupilstop_tag}_{self.dm_tag}_D{D:.2f}_pad{pad}')

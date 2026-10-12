"""ErrorBudgetGenerator: runs the SPECULA calibrations and post-processes them into the products read by the budget machine
(layout: naming.py; the SPECULA ymls are in config/). SPECULA is driven as in LBTSynIM: a nested override dict {block: {param: value}}
is written as `<block>_override` YAML and passed to `specula main.yml extra.yml overrides.yml`; every independent atmospheric
realisation is its own SPECULA run with `atmo_random.seed = seed0 + i + 1` and the results are averaged (IMs) or pooled (aliasing runs).
Steps (existing products are skipped unless overwrite=True): pupil data; aliasing / slope null / throughput / noise illumination;
diffraction-limited IM; reconstructors; aliasing modal signal; IM with N modes perfectly corrected; optionally IMs with a partial
correction (calib_pc.yml serves both: only the correction vector differs). The sensing source and the pyramid always get the same
wavelength; the science source is propagated like the NGS."""
import logging
import os
import os.path as op
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import yaml

from . import utils
from .naming import Paths, snap_seeing
from .noise import PupilGeometry, pixel_fractions, slope_variance_diag
from .reconstructors import compute_reconstructor, read_rec, save_rec

HERE = op.dirname(op.abspath(__file__))
log = logging.getLogger('aoeb.generation')


def _plain(v):
    """numpy scalars / tuples -> plain Python (yaml.safe_dump)."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


def merge(*dicts):
    """Deep merge of nested override dicts (later wins)."""
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_overrides_yaml(overrides, path):
    """{block: {param: value}} -> `<path>.yml` with `<block>_override` keys (the SPECULA merge convention)."""
    with open(f'{path}.yml', 'w') as f:
        yaml.safe_dump({f'{k}_override': _plain(v) for k, v in overrides.items()}, f, sort_keys=False, default_flow_style=False)


def run_specula(main_yaml, override_path, extra_yaml=None):
    """extra_yaml: a yml path or a list of them (appended in order, before the override file)."""
    extra = [] if not extra_yaml else [extra_yaml] if isinstance(extra_yaml, (str, os.PathLike)) else list(extra_yaml)
    subprocess.run(['specula', str(main_yaml), *map(str, extra), f'{override_path}.yml'], check=True)


# ----------------------------------------------------------------------------- correction vectors
def _radial_order(n):
    try:
        from specula.mmlib.utils import radial_order      # the function of the reference save_correction_vector
    except ImportError:
        from .utils import radial_order
    return int(radial_order(n))


def correction_vector(max_corr, min_corr, Nmodes, Ncorrmodes=None):
    """Per-mode correction factors (1 = corrected, 0 = uncorrected): max_corr (tip-tilt group) -> min_corr (highest group), linear in the
    radial order, the same factor for all the modes of an order; modes up to Ncorrmodes that no order covers get min_corr; modes beyond
    Ncorrmodes get 0. max_corr = min_corr = 1 is the perfect correction."""
    if Ncorrmodes is None:
        Ncorrmodes = Nmodes
    if not 0.0 <= min_corr <= max_corr <= 1.0:
        raise ValueError(f'need 0 <= min_corr <= max_corr <= 1, got {min_corr}, {max_corr}')
    max_rad_order = _radial_order(Nmodes) + 1
    if max_rad_order < 4:
        raise ValueError(f'too few modes ({Nmodes}) for a correction vector')
    leakage_per_order = np.linspace(1.0 - max_corr, 1.0 - min_corr, max_rad_order - 2)
    cc = 1.0 - leakage_per_order
    tt = np.hstack([np.repeat(cc[i - 2], i) for i in range(2, max_rad_order)])
    residuals = np.zeros(Nmodes)
    length_to_fill = min(len(tt), Ncorrmodes)
    residuals[:length_to_fill] = tt[:length_to_fill]
    if Ncorrmodes > length_to_fill:
        residuals[length_to_fill:Ncorrmodes] = min_corr
    return residuals


def save_correction_vector(root_dir, max_corr, min_corr, Nmodes, Ncorrmodes=None):
    """Write the correction vector as a SPECULA BaseValue file <root_dir>/data/correction_vector_{Ncorrmodes}modes_c{max}-{min}.fits
    (header VERSION, OBJ_TYPE = BaseValue, NDARRAY = 1, MAX_CORR, MIN_CORR); returns the file stem. One function for the perfect
    correction (max_corr = min_corr = 1) and for partial corrections. The vector multiplies the reconstructed (modal-analysis) modes
    that command the DM, which SPECULA applies as a correction (negative sign): 1 = perfectly corrected mode, 0 = uncorrected."""
    from astropy.io import fits
    if Ncorrmodes is None:
        Ncorrmodes = Nmodes
    vec = correction_vector(max_corr, min_corr, Nmodes, Ncorrmodes)
    fname = f'correction_vector_{Ncorrmodes}modes_c{max_corr:1.2f}-{min_corr:1.2f}'
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['OBJ_TYPE'] = 'BaseValue'
    hdr['NDARRAY'] = 1
    hdr['MAX_CORR'] = max_corr
    hdr['MIN_CORR'] = min_corr
    utils.write_fits(op.join(root_dir, 'data', fname + '.fits'), vec, header=hdr)
    log.info('saved correction vector %s', fname)
    return fname


# ----------------------------------------------------------------------------- post-processing
def _load_run(scratch, discard):
    slopes, frames, turb = (utils.read_fits(op.join(scratch, f'{n}.fits')) for n in ('slopes', 'pyr_frames', 'turb_modes'))
    if not np.any(slopes):
        raise ValueError(f'{scratch}: all the slopes are zero (the SPECULA run produced nothing usable)')
    first = int(np.argmax(np.any(slopes != 0, axis=1)))               # leading all-zero frames are start-up artefacts
    start = max(discard, first)
    return slopes[start:], frames[start:], turb[start:]


def check_independent(turb_modes, tol=0.5):
    """Warn if consecutive frames are strongly correlated (repeated or held atmospheric screens)."""
    t = np.asarray(turb_modes, float)[:, :10]
    r = [abs(np.corrcoef(t[:-1, k], t[1:, k])[0, 1]) for k in range(t.shape[1]) if t[:, k].std() > 0]
    if not r or np.nanmedian(r) > tol:
        warnings.warn('consecutive aliasing-run frames are not independent (median lag-1 correlation '
                      f'{np.nanmedian(r) if r else 1.0:.2f}): check atmo_random.update_interval and seed')


def process_aliasing_runs(paths, rMod, n_subap, seeing, N, scratch_dirs, discard=0):
    """Pool the data-store output of the independent aliasing runs into the products; `slopes` (the "done" marker) is written last."""
    runs = [_load_run(d, discard) for d in scratch_dirs]
    slopes, turb = np.concatenate([r[0] for r in runs]), np.concatenate([r[2] for r in runs])
    counts = np.array([len(r[1]) for r in runs], float)
    avg = sum(r[1].mean(axis=0) * c for r, c in zip(runs, counts)) / counts.sum()
    check_independent(turb)
    thrp = pixel_fractions(avg, PupilGeometry.from_file(paths.pupdata(n_subap), n_subap, avg.shape[0]))[1]
    utils.write_fits(paths.turbmodes(rMod, n_subap, seeing, N), turb.astype(np.float32))
    utils.write_fits(paths.avgframe(rMod, n_subap, seeing, N), avg)
    utils.write_fits(paths.sn(rMod, n_subap, seeing, N), slopes.mean(axis=0))
    utils.write_fits(paths.thrp(rMod, n_subap, seeing, N), np.array([thrp]))
    utils.write_fits(paths.slopes(rMod, n_subap, seeing, N), slopes.astype(np.float32))
    return thrp


def compute_alias_product(paths, rMod, n_subap, seeing, N_rec, N_corr):
    """std(R_Nrec @ slopes_Ncorr) per mode: the aliasing seen by the sensor controlling N_rec modes."""
    rec = read_rec(paths.rec(rMod, n_subap, N_rec))
    slopes = utils.read_fits(paths.slopes(rMod, n_subap, seeing, N_corr)).astype(float)
    alias = np.std(rec @ slopes.T, axis=1)
    utils.write_fits(paths.alias(rMod, n_subap, seeing, N_rec, N_corr), alias)
    return alias


# ----------------------------------------------------------------------------- configuration
@dataclass
class GeneratorConfig:
    root_dir: str
    config_dir: str = op.join(HERE, 'config')      # relative yml names are looked up here (default: the packaged ymls)
    main_config: str = 'ristretto_main.yml'
    calib_pupdata: str = 'calib_pupdata.yml'
    calib_im: str = 'calib_im.yml'
    calib_alias_sn_thrp: str = 'calib_alias_sn_thrp.yml'
    calib_pc: str = 'calib_pc.yml'                 # perfect and partial correction: only the correction vector differs
    diameter: float = 8.2                          # telescope diameter [m]: main.pixel_pitch = diameter / pixel_pupil
    pixel_pupil: int = None                        # main.pixel_pupil: must match the pupilstop and the DM influence functions;
                                                   # None: the size of the pupilstop array
    wavelengths_nm: tuple = (750.0,)               # sensing wavelengths: pyr.wavelengthInNm AND source_ngs.wavelengthInNm, always equal
    science_nm: float = 1650.0                     # source_science.wavelengthInNm (propagated like the NGS)
    rMods: tuple = (0.0, 1.0, 2.0, 3.0)
    n_subaps: tuple = (10, 20, 40, 48)
    modes_per_subap: dict = field(default_factory=lambda: {10: 54, 20: 250, 40: 1000, 48: 1200})
    n_modes_ho: int = 1200
    n_modes_max: int = 1821                        # columns of the diffraction-limited IM
    seeings: tuple = (0.5, 0.7, 0.9, 1.1, 1.3)
    time_step: float = 0.0005
    alias_magnitude: float = 5.0
    n_screens_alias: int = 1                       # independent aliasing runs (seeds seed0 + 1 ...), pooled
    n_screens_pc: int = 70                         # independent PC-IM runs (one push-pull cycle and one screen each), averaged
    pc_corrections: tuple = ()                     # ((max_corr, min_corr), ...): extra partial-correction IMs (save_correction_vector)
    seed0: int = 0
    discard_frames: int = 0                        # frames dropped from the aliasing runs (leading all-zero frames always are)
    L0: float = 25.0                               # outer scale of the aliasing and PC runs (the budget's L0)
    pup_dist_min: float = 16.0
    pup_dist_per_subap: float = 1.25               # pyr.pup_dist = max(pup_dist_min, pup_dist_per_subap * n_subap)
    pupilstop_tag: str = 'vlt_pupil_160pixels'
    dm_tag: str = 'bmc2k_vlt'                      # DM data objects {dm_tag}_ifunc, {dm_tag}_m2c, {dm_tag}_kl_inv
    overwrite: bool = False


STEPS = {'pupdata': 'calibrate_pupdata', 'alias': 'calibrate_alias_sn_thrp', 'im_dl': 'calibrate_im_dl', 'recs': 'compute_recs',
         'aliases': 'compute_aliases', 'pc': 'calibrate_im_pc', 'pc_levels': 'calibrate_im_pc_levels'}


class ErrorBudgetGenerator:
    def __init__(self, cfg: GeneratorConfig):
        wls = tuple(cfg.wavelengths_nm) if cfg.wavelengths_nm is not None else ()
        if not wls or any(w is None for w in wls):
            raise ValueError('wavelengths_nm must list the sensing wavelengths [nm]: the pyramid and its source need one (and the same)')
        levels = [(float(a), float(b)) for a, b in cfg.pc_corrections]
        for a, b in levels:
            if not 0.0 < b <= a <= 1.0:
                raise ValueError(f'pc_corrections: need 0 < min_corr <= max_corr <= 1, got ({a}, {b})')
        if len({f'{a:1.2f}-{b:1.2f}' for a, b in levels}) != len(levels):
            raise ValueError('pc_corrections: two levels have the same name (resolution 0.01)')
        self.cfg = cfg
        self.paths = Paths(cfg.root_dir, cfg.pupilstop_tag, cfg.dm_tag)       # wavelength-independent files
        self.runs = []                                                        # {main, extra, overrides} of every SPECULA call
        self._npix = None

    # ---- geometry / bookkeeping
    def pup_dist(self, n_subap):
        return max(self.cfg.pup_dist_min, self.cfg.pup_dist_per_subap * n_subap)

    def n_modes(self, n_subap):
        return self.cfg.modes_per_subap[n_subap]

    def corrected_sets(self, n_subap):
        """Numbers of perfectly corrected modes needed for a sensor: standard and multistage."""
        return sorted({self.n_modes(n_subap), self.cfg.n_modes_ho})

    def seeds(self, n):
        return [self.cfg.seed0 + i + 1 for i in range(n)]

    def wavelengths(self):
        return sorted({round(float(w)) for w in self.cfg.wavelengths_nm})

    def _sensors(self):
        """(paths, n_subap, rMod) for every wavelength (rounded to the tag resolution, see naming.py)."""
        c = self.cfg
        for wl, n, rMod in product(self.wavelengths(), c.n_subaps, sorted({round(float(r), 1) for r in c.rMods})):
            yield self.paths.at(wl), n, rMod

    def _cases(self):
        """(paths, n_subap, rMod, seeing, N): every sensor x seeing x corrected-mode set."""
        seeings = sorted({snap_seeing(s) for s in self.cfg.seeings})
        for P, n, rMod in self._sensors():
            for seeing, N in product(seeings, self.corrected_sets(n)):
                yield P, n, rMod, seeing, N

    def _needed(self, path):
        return self.cfg.overwrite or not op.exists(path)

    def _yml(self, name):
        return name if op.isabs(name) else op.join(self.cfg.config_dir, name)

    def _pixel_pupil(self):
        """main.pixel_pupil: the configured value, else the size of the pupilstop array (as the budget machine reads it)."""
        if self.cfg.pixel_pupil:
            return int(self.cfg.pixel_pupil)
        if self._npix is None:
            pupil = utils.read_fits(self.paths.pupilstop())
            if pupil.ndim != 2 or pupil.shape[0] != pupil.shape[1]:
                raise ValueError(f'the pupilstop {self.paths.pupilstop()} is not a square array: {pupil.shape}')
            self._npix = int(pupil.shape[0])
        return self._npix

    def check_inputs(self):
        """The static inputs of the budget machine must exist before thousands of SPECULA runs are started."""
        missing = [p for p in (self.paths.pupilstop(), self.paths.kl_inv()) if not op.exists(p)]
        if missing:
            raise FileNotFoundError('missing inputs: ' + ', '.join(missing))

    # ---- SPECULA
    def _launch(self, main, extra, overrides):
        """The only place where SPECULA is started: a unique temporary override file per run (parallel safe)."""
        scratch = op.join(self.cfg.root_dir, 'scratch')
        os.makedirs(scratch, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            path = op.join(tmp, 'overrides')
            write_overrides_yaml(overrides, path)
            run_specula(main, path, extra)

    def _simulate(self, extra_ymls, overrides):
        """One SPECULA run: main yml + extra yml(s) + overrides. Always set: main.root_dir, main.pixel_pupil and main.pixel_pitch (they
        must match the pupilstop and the influence functions, or the electric fields of the atmosphere and of the DM differ in size)
        and pupilstop.tag."""
        c = self.cfg
        npix = self._pixel_pupil()
        ov = merge({'main': {'root_dir': str(c.root_dir), 'pixel_pupil': npix, 'pixel_pitch': c.diameter / npix},
                    'pupilstop': {'tag': c.pupilstop_tag}}, overrides)
        main = self._yml(c.main_config)
        extra = [self._yml(y) for y in ([extra_ymls] if isinstance(extra_ymls, str) else extra_ymls)]
        self.runs.append(dict(main=main, extra=extra, overrides=ov))
        self._launch(main, extra, ov)

    def _pyr(self, P, rMod, n_subap):
        """Pyramid geometry; the pyramid and its source always have the same wavelength."""
        wl = float(P.wavelength_nm)
        return {'pyr': {'pup_diam': float(n_subap), 'pup_dist': float(self.pup_dist(n_subap)), 'mod_amp': float(rMod), 'wavelengthInNm': wl},
                'source_ngs': {'wavelengthInNm': wl},
                'pyr_slopes': {'pupdata_object': f'pyr_pupdata_{n_subap:.0f}x{n_subap:.0f}'}}

    def _dm(self, P, *blocks):
        return {b: {'ifunc_object': P.ifunc_name, 'm2c_object': P.m2c_name} for b in blocks}

    # ---- steps
    def calibrate_pupdata(self):
        """1. pupil data (wavelength independent: made at the first sensing wavelength)."""
        P = self.paths.at(self.wavelengths()[0])
        for n in self.cfg.n_subaps:
            if self._needed(P.pupdata(n)):
                wl = float(P.wavelength_nm)
                self._simulate(self.cfg.calib_pupdata, {
                    'pyr': {'pup_diam': float(n), 'pup_dist': float(self.pup_dist(n)), 'wavelengthInNm': wl},    # modulation as in the main yml
                    'source_ngs': {'wavelengthInNm': wl}, 'pyr_pupdata': {'output_tag': f'pyr_pupdata_{n:.0f}x{n:.0f}'}})

    def calibrate_alias_sn_thrp(self):
        """2. aliasing + slope null + throughput + noise illumination: N modes perfectly corrected, one run per seed, pooled; the
        science source goes through the same screens and DM."""
        c = self.cfg
        cases = list(self._cases())
        for k, (P, n, rMod, seeing, N) in enumerate(cases, 1):
            if not self._needed(P.slopes(rMod, n, seeing, N)):
                continue
            log.info('[%d/%d] aliasing run %s', k, len(cases), P.sim(rMod, n, seeing, N))
            sim = op.join(c.root_dir, 'scratch', P.sim(rMod, n, seeing, N))
            seeds = self.seeds(c.n_screens_alias)
            dirs = [op.join(sim, f'seed{seed}') for seed in seeds]
            for seed, d in zip(seeds, dirs):
                self._simulate(c.calib_alias_sn_thrp, merge(self._pyr(P, rMod, n), self._dm(P, 'dm_perfect'), {
                    'atmo_random': {'seed': seed, 'L0': c.L0}, 'seeing_random': {'constant': float(seeing)},
                    'dm_perfect': {'nmodes': N}, 'modal_analysis': {'ifunc_inv_object': P.kl_inv_name, 'nmodes': N},
                    'source_ngs': {'magnitude': float(c.alias_magnitude)}, 'source_science': {'wavelengthInNm': float(c.science_nm)},
                    'data_store': {'store_dir': d, 'create_tn': False}}))
            process_aliasing_runs(P, rMod, n, seeing, N, dirs, c.discard_frames)
            shutil.rmtree(sim, ignore_errors=True)

    def calibrate_im_dl(self):
        """3. diffraction-limited IM."""
        c = self.cfg
        for P, n, rMod in self._sensors():
            path = P.im_dl(rMod, n)
            if self._needed(path):
                log.info('diffraction-limited IM %s', op.basename(path))
                self._simulate(c.calib_im, merge(self._pyr(P, rMod, n), self._dm(P, 'dm'), {
                    'pyr_im_calibrator': {'im_tag': op.basename(path)[:-5], 'nmodes': c.n_modes_max, 'overwrite': True},
                    'pushpull': {'nmodes': c.n_modes_max}, 'dm': {'nmodes': c.n_modes_max},
                    'main': {'total_time': c.n_modes_max * 2 * c.time_step}}))

    def compute_recs(self):
        """4. SVD reconstructors of the first N modes of the diffraction-limited IM."""
        for P, n, rMod in self._sensors():
            N, path = self.n_modes(n), P.rec(rMod, n, self.n_modes(n))
            if self._needed(path):
                save_rec(path, compute_reconstructor(utils.read_fits(P.im_dl(rMod, n)), N))

    def compute_recs_weighted(self, method, flux, fs, detector, seeing, machine, slopes_from_intensity=False):
        """ML or MAP reconstructors matched to a flux, rate, detector and seeing."""
        n_phot, seeing = detector.n_det(flux, fs), snap_seeing(seeing)
        for P, n, rMod in self._sensors():
            N = self.n_modes(n)
            path = P.rec(rMod, n, N, method, n_phot, seeing)
            if not self._needed(path):
                continue
            frame = utils.read_fits(P.avgframe(rMod, n, seeing, N))
            cn = slope_variance_diag(frame, PupilGeometry.from_file(self.paths.pupdata(n), n, frame.shape[0]), n_phot, detector,
                                     slopes_from_intensity)
            ca = machine.turbulence(seeing, N)[0] if method == 'map' else None
            save_rec(path, compute_reconstructor(utils.read_fits(P.im_dl(rMod, n)), N, method, cn_diag=cn, ca_diag=ca),
                     method=method, n_phot=float(n_phot), ron=detector.ron_e, f2=detector.f2, seeing=seeing)

    def compute_aliases(self):
        """5. aliasing modal signal std(R @ slopes) for the standard and the multistage case."""
        for P, n, rMod, seeing, Nc in self._cases():
            if self._needed(P.alias(rMod, n, seeing, self.n_modes(n), Nc)):
                compute_alias_product(P, rMod, n, seeing, self.n_modes(n), Nc)

    def _pc_average(self, P, n, rMod, seeing, N, path, vector):
        """n_screens_pc independent runs (LBTSynIM style): one push-pull cycle (2 N frames) and one screen per run, seed per run;
        the IMs are averaged into `path`. `vector`: stem of the correction vector."""
        c = self.cfg
        accum = None
        for seed in self.seeds(c.n_screens_pc):
            self._simulate(c.calib_pc, merge(self._pyr(P, rMod, n), self._dm(P, 'dm', 'dm_random'), {
                'main': {'total_time': N * 2 * c.time_step},
                'atmo_random': {'update_interval': N * 2, 'seed': seed, 'L0': c.L0}, 'seeing_random': {'constant': float(seeing)},
                'modal_analysis_random': {'ifunc_inv_object': P.kl_inv_name, 'nmodes': N},
                'scale_random': {'constant_mul_data': vector},
                'dm_random': {'nmodes': N}, 'dm': {'nmodes': N}, 'pushpull': {'nmodes': N, 'ncycles': 1},
                'pyr_im_calibrator': {'im_tag': op.basename(path)[:-5], 'nmodes': N, 'overwrite': True}}))
            im = utils.read_fits(path)
            accum = im if accum is None else accum + im
        utils.write_fits(path, accum / c.n_screens_pc)

    def calibrate_im_pc(self):
        """6. IM with N modes perfectly corrected (correction vector of ones)."""
        cases = list(self._cases())
        for k, (P, n, rMod, seeing, N) in enumerate(cases, 1):
            path = P.im_pc(rMod, n, seeing, N)
            if self._needed(path):
                log.info('[%d/%d] PC IM (perfect) %s', k, len(cases), op.basename(path))
                self._pc_average(P, n, rMod, seeing, N, path, save_correction_vector(self.cfg.root_dir, 1.0, 1.0, N))

    def calibrate_im_pc_levels(self):
        """6b. IMs with a partial correction, one per (max_corr, min_corr) of `pc_corrections`: the correction vector decreases from
        max_corr at the tip-tilt to min_corr at the highest order (save_correction_vector)."""
        cases = list(self._cases())
        levels = sorted({(float(a), float(b)) for a, b in self.cfg.pc_corrections if (a, b) != (1.0, 1.0)})
        for k, (P, n, rMod, seeing, N) in enumerate(cases, 1):
            for level in levels:
                path = P.im_pc(rMod, n, seeing, N, level)
                if self._needed(path):
                    log.info('[%d/%d] PC IM c%1.2f-%1.2f %s', k, len(cases), level[0], level[1], op.basename(path))
                    self._pc_average(P, n, rMod, seeing, N, path, save_correction_vector(self.cfg.root_dir, *level, N))

    def run(self, steps=None):
        """The steps in their natural order (names in STEPS; None: all, `pc_levels` only if pc_corrections is not empty)."""
        names = list(STEPS) if steps is None else list(steps)
        unknown = [s for s in names if s not in STEPS]
        if unknown:
            raise ValueError(f'unknown steps {unknown}; choose from {list(STEPS)}')
        if 'pc_levels' in names and not self.cfg.pc_corrections:
            if steps is not None:
                raise ValueError('step pc_levels needs generation.pc_corrections')
            names.remove('pc_levels')
        self.check_inputs()
        for name in STEPS:
            if name in names:
                log.info('step %s', name)
                getattr(self, STEPS[name])()

    def run_all(self):
        self.run()

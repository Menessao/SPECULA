"""One YAML file for a whole study: Config.load(path, overrides) -> machine / generator / chromatic model / scan.

config/defaults.yml is the single source of the defaults and documents every key; unknown keys are errors; overrides are
'section.key=value' (value parsed as YAML, floats like 1e6 / 5e-4 included); the resolved configuration can be dumped next to the results."""
import copy
import os.path as op
import re

import numpy as np
import yaml

from .chromatic import ChromaticModel
from .utils import effective_wind

HERE = op.dirname(op.abspath(__file__))


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e6, 5e-4 and 1.0e6 as floats (YAML 1.1 wants a dot and a signed exponent)."""


_Loader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?|[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)'
               r'|\.[0-9_]+(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$'), list('-+0123456789.'))


def _load(text):
    return yaml.load(text, _Loader)


with open(op.join(HERE, 'config', 'defaults.yml')) as _f:
    DEFAULTS = _load(_f.read())
_REPLACE = {'modes_per_subap'}                      # dicts that are replaced, not merged


def _merge(base, new, path=''):
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"unknown config key '{path}{k}'; valid keys: {sorted(base)}")
        if isinstance(base[k], dict) and isinstance(v, dict) and k not in _REPLACE:
            _merge(base[k], v, f'{path}{k}.')
        else:
            base[k] = v


def _nest(dotted, value):
    for k in reversed(dotted.split('.')):
        value = {k: value}
    return value


class Config:
    def __init__(self, data):
        self.d = data

    @classmethod
    def load(cls, path=None, overrides=()):
        d = copy.deepcopy(DEFAULTS)
        if path:
            with open(path) as f:
                _merge(d, _load(f.read()) or {})
        for o in overrides:
            key, value = o.split('=', 1)
            _merge(d, _nest(key.strip(), _load(value)))
        atm = d['atmosphere']
        if atm['from_specula']:
            p = atm['from_specula']
            with open(p if op.isabs(p) else op.join(d['paths']['specula_dir'] or op.join(HERE, 'config'), p)) as f:
                y = _load(f.read())
            a = y['atmo']
            for key, names in (('heights', ('heights',)), ('cn2', ('Cn2', 'cn2'))):
                found = [n for n in names if n in a]
                if not found:
                    raise KeyError(f"'{key}' not found in the atmo section of {p} (keys: {sorted(a)})")
                atm[key] = a[found[0]]
            ws = [n for n in ('wind_speed', 'windSpeed') if n in a]
            if ws:                                                 # inside atmo, or the top-level WaveGenerator block `wind_speed`
                atm['wind_speed'] = a[ws[0]]
            elif 'constant' in y.get('wind_speed', {}):
                atm['wind_speed'] = y['wind_speed']['constant']
            else:
                raise KeyError(f"wind_speed not found in {p}: neither atmo.wind_speed nor a wind_speed block with `constant`")
        if not len(atm['heights']) == len(atm['cn2']) == len(atm['wind_speed']):
            raise ValueError('atmosphere heights, cn2 and wind_speed must have the same length')
        return cls(d)

    def dump(self, path):
        with open(path, 'w') as f:
            yaml.safe_dump(self.d, f, sort_keys=False)

    # ---- derived quantities
    def wind(self):
        a = self.d['atmosphere']
        return float(a['wind']) if a['wind'] is not None else effective_wind(a['wind_speed'], a['cn2'])

    def sensing(self):
        w = self.d['wavelengths']['sensing_nm']
        return (None,) if w is None else tuple(np.atleast_1d(w).astype(float))

    @staticmethod
    def _range(r, log):
        if isinstance(r, dict):
            return (np.geomspace if log else np.linspace)(r['min'], r['max'], int(r['n']))
        return np.atleast_1d(np.asarray(r, float))

    def fluxes(self):
        return self._range(self.d['scan']['flux'], True)

    def freqs(self):
        return self._range(self.d['loop']['fs'], False)

    # ---- objects
    def _root(self):
        if not self.d['paths']['root_dir']:
            raise ValueError('paths.root_dir is required')
        return self.d['paths']['root_dir']

    def detector(self):
        from .noise import Detector
        return Detector(**self.d['detector'])

    def machine(self, **kw):
        from .budget import ErrorBudgetMachine
        p, t, s, o = (self.d[k] for k in ('paths', 'telescope', 'sensors', 'options'))
        return ErrorBudgetMachine(self._root(), diameter=t['diameter'], L0=t['L0'], wind=self.wind(), detector=self.detector(),
                                  pupilstop_tag=p['pupilstop_tag'], dm_tag=p['dm_tag'], modes_per_subap=s['modes_per_subap'],
                                  n_modes_ho=s['n_modes_ho'], kl_pad=o['kl_pad'], slopes_from_intensity=s['slopes_from_intensity'],
                                  asm=o['asm'], max_sensitivity=self.d['loop']['max_sensitivity'], optimize_freq=o['optimize_freq'],
                                  optimize_modes=o['optimize_modes'], freq_min=o['freq_min'], **kw)

    def iir(self):
        """IirControllers of loop.iir_file (a file saved by SPECULA's IirFilterData); loop.iir_optimize_gain: optimise the global gain."""
        from .filter_optimizers import IirControllers
        lp = self.d['loop']
        if not lp['iir_file']:
            raise ValueError("controller 'iir' needs loop.iir_file")
        return IirControllers.from_fits(lp['iir_file'], optimize_gain=bool(lp['iir_optimize_gain']))

    def psf_engine(self, machine=None, pad=4):
        from .psf import PsfEngine
        return PsfEngine(machine or self.machine(), pad)

    def chromatic(self):
        c, a = self.d['chromatic'], self.d['atmosphere']
        if not c['enabled']:
            return None
        return ChromaticModel(a['heights'], a['cn2'], self.d['wavelengths']['science_nm'], **{k: v for k, v in c.items() if k != 'enabled'})

    def generator(self):
        from .generation import ErrorBudgetGenerator, GeneratorConfig
        p, s, g = self.d['paths'], self.d['sensors'], self.d['generation']
        kw = dict(root_dir=self._root(), diameter=self.d['telescope']['diameter'], main_config=p['main'], calib_pupdata=p['pupdata'], calib_im=p['im'],
                  calib_alias_sn_thrp=p['alias'], calib_pc=p['pc'], wavelengths_nm=self.sensing(),
                  science_nm=self.d['wavelengths']['science_nm'], rMods=tuple(s['rMods']),
                  n_subaps=tuple(s['n_subaps']), modes_per_subap=dict(s['modes_per_subap']), n_modes_ho=s['n_modes_ho'],
                  seeings=tuple(self.d['scan']['seeings']), pupilstop_tag=p['pupilstop_tag'], dm_tag=p['dm_tag'],
                  L0=self.d['telescope']['L0'], **g)
        if p['specula_dir']:
            kw['config_dir'] = p['specula_dir']
        kw['pc_corrections'] = tuple(tuple(x) for x in kw['pc_corrections'])
        return ErrorBudgetGenerator(GeneratorConfig(**kw))

    def budget_kwargs(self):
        lp, o = self.d['loop'], self.d['options']
        return dict(latency_s=lp['latency_s'], n_fold=lp['n_fold'], crosstalk=o['crosstalk'], fixed_point=o['fixed_point'],
                    chromatic=self.chromatic())

    def scan(self, machine=None, progress=True):
        """Full scan of the configuration -> (modes_df, totals_df)."""
        m = machine or self.machine()
        s, lp = self.d['scan'], self.d['loop']
        k = s['store_modes']
        return m.scan(s['seeings'], self.d['sensors']['rMods'], self.d['sensors']['n_subaps'], self.fluxes(), self.freqs(),
                      tuple(lp['scenarios']), tuple(self.iir() if c == 'iir' else c for c in lp['controllers']), store_modes=list(range(k)) if k else None,
                      progress=progress, wavelengths=self.sensing(), **self.budget_kwargs())

"""One-off: regenerate the 850 nm aliasing products of the VLT root (slopes, turbmodes, avgframe, sn, thrp, alias). The stored ones (2026-10-23)
predate the generator update of 2026-10-27: their slopes do not match a fresh run (same seed), while the 1250 nm ones are reproduced exactly.
They are generated in <root>/_regen_850 (resumable), and only when the whole set is there are the old files moved to <root>/_stale_850_20261023
and the new ones moved into place, so the old products stay usable until then."""
import glob
import os
import os.path as op
import shutil
import sys

sys.path.insert(0, op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']
from aoeb.generation import ErrorBudgetGenerator, GeneratorConfig  # noqa: E402

V = '/raid1/mmenessini/calibration/VLT'
R = op.join(V, '_regen_850')
DIRS = ('aliasing', 'frames', 'slopenulls', 'throughput')
os.makedirs(R, exist_ok=True)
for d in ('pupilstop', 'ifunc', 'm2c', 'pupils', 'rec', 'phasescreens'):
    if op.exists(op.join(V, d)) and not op.exists(op.join(R, d)):
        os.symlink(op.join(V, d), op.join(R, d))
gen = ErrorBudgetGenerator(GeneratorConfig(
    root_dir=R, main_config='ristretto_vis.yml', wavelengths_nm=(850.0,), science_nm=1650.0, rMods=(0.0, 0.5, 1.0, 2.0, 3.0),
    n_subaps=(10, 24, 36, 48), modes_per_subap={10: 54, 24: 280, 36: 800, 48: 1200}, n_modes_ho=1200, seeings=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
    pupilstop_tag='vlt_pupil_220pixels', dm_tag='bmc2k_vlt', L0=25.0))
gen.run(['alias', 'aliases'])
new = {d: sorted(glob.glob(op.join(R, d, '*wvl850*'))) for d in DIRS}
n_slopes = sum(f.endswith('_slopes.fits') for f in new['aliasing'])
n_alias = sum(f.endswith('_alias.fits') for f in new['aliasing'])
print('generated:', {d: len(v) for d, v in new.items()}, 'slopes', n_slopes, 'alias', n_alias, flush=True)
if (n_slopes, n_alias) != (210, 210) or any(len(new[d]) != 210 for d in DIRS[1:]):
    sys.exit('incomplete set: nothing swapped')
stale = op.join(V, '_stale_850_20261023')
for d in DIRS:
    os.makedirs(op.join(stale, d), exist_ok=True)
    for f in new[d]:
        target = op.join(V, d, op.basename(f))
        if op.exists(target):
            shutil.move(target, op.join(stale, d, op.basename(f)))
        shutil.move(f, target)
print('swapped; old files in', stale, flush=True)

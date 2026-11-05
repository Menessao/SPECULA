"""One-off: regenerate every 750 nm product of the VLT root with the current chain (dm_tag bmc2k_vlt, 1821-mode IM). The stored ones (2026-10-12..17)
predate it: they use the older DM dm1200_220pix (the 48x48 IM is reproduced exactly with it, so that set is valid for that DM, IM of 1359 modes),
and 10x10 matches neither DM (other pupil data). Generated in <root>/_regen_750 (resumable); only when the whole
set is there are the old files moved to <root>/_stale_750_20261012 and the new ones moved into place. PC IMs: 70 SPECULA runs per case, ~1 week."""
import glob
import os
import os.path as op
import shutil
import sys

sys.path.insert(0, op.dirname(op.dirname(op.dirname(op.abspath(__file__)))))
os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']
from aoeb.generation import ErrorBudgetGenerator, GeneratorConfig  # noqa: E402

V = '/raid1/mmenessini/calibration/VLT'
R = op.join(V, '_regen_750')
EXPECTED = {'aliasing': 420, 'frames': 140, 'slopenulls': 140, 'throughput': 140, 'im': 156, 'rec': 16}   # aliasing: slopes + turbmodes + alias
os.makedirs(R, exist_ok=True)
for d in ('pupilstop', 'ifunc', 'm2c', 'pupils', 'phasescreens'):
    if op.exists(op.join(V, d)) and not op.exists(op.join(R, d)):
        os.symlink(op.join(V, d), op.join(R, d))
gen = ErrorBudgetGenerator(GeneratorConfig(
    root_dir=R, main_config='ristretto_vis.yml', wavelengths_nm=(750.0,), science_nm=1650.0, rMods=(0.0, 1.0, 2.0, 3.0), n_subaps=(10, 20, 40, 48),
    modes_per_subap={10: 54, 20: 250, 40: 1000, 48: 1200}, n_modes_ho=1200, n_modes_max=1821, seeings=(0.5, 0.7, 0.9, 1.1, 1.3),
    pupilstop_tag='vlt_pupil_220pixels', dm_tag='bmc2k_vlt', L0=25.0))
gen.run(['alias', 'im_dl', 'recs', 'aliases', 'pc'])
new = {d: sorted(glob.glob(op.join(R, d, '*wvl750*'))) for d in EXPECTED}
print('generated:', {d: len(v) for d, v in new.items()}, flush=True)
if any(len(new[d]) != n for d, n in EXPECTED.items()):
    sys.exit(f'incomplete set (expected {EXPECTED}): nothing swapped')
stale = op.join(V, '_stale_750_20261012')
for d, files in new.items():
    os.makedirs(op.join(stale, d), exist_ok=True)
    for f in files:
        target = op.join(V, d, op.basename(f))
        if op.exists(target):
            shutil.move(target, op.join(stale, d, op.basename(f)))
        shutil.move(f, target)
print('swapped; old files in', stale, flush=True)

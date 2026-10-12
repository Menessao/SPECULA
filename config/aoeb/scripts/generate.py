"""Run the SPECULA calibrations of the error budget (aoeb.generation) from the command line.

    python -m aoeb.scripts.generate --root /raid1/RISTRETTO                      # all steps, packaged ymls and defaults
    python -m aoeb.scripts.generate study.yml sensors.n_subaps=[48] --steps im_dl recs
    aoeb-generate study.yml wavelengths.sensing_nm=[750,1650] generation.pc_corrections=[[0.99,0.2]]

The optional study file and the `section.key=value` overrides are those of the configuration (config/defaults.yml lists every key).
Steps, always run in this order: pupdata, alias, im_dl, recs, aliases, pc, pc_levels (the last only with generation.pc_corrections).
Products that already exist are skipped unless --overwrite; every SPECULA run is logged."""
import argparse
import logging
import sys

from ..config import Config
from ..generation import STEPS


def main(argv=None):
    p = argparse.ArgumentParser(prog='aoeb-generate', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('args', nargs='*', help='[study.yml] [section.key=value ...]')
    p.add_argument('--root', help='products root (paths.root_dir)')
    p.add_argument('--steps', nargs='+', choices=list(STEPS), help='steps to run (default: all)')
    p.add_argument('--overwrite', action='store_true', help='redo products that already exist')
    p.add_argument('--list-steps', action='store_true', help='print the steps and exit')
    p.add_argument('-q', '--quiet', action='store_true', help='log only warnings')
    a = p.parse_args(argv)
    if a.list_steps:
        print('\n'.join(f'{k:10s} {v}' for k, v in STEPS.items()))
        return 0
    overrides = [x for x in a.args if '=' in x]
    files = [x for x in a.args if '=' not in x]
    if len(files) > 1:
        p.error(f'at most one study file, got {files}')
    if a.root:
        overrides.append(f'paths.root_dir={a.root}')
    if a.overwrite:
        overrides.append('generation.overwrite=true')
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        stream=sys.stdout)
    cfg = Config.load(files[0] if files else None, overrides)
    gen = cfg.generator()
    gen.run(a.steps)
    print(f'done: {len(gen.runs)} SPECULA runs')
    return 0


if __name__ == '__main__':
    sys.exit(main())

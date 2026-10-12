"""python -m aoeb.scripts.run {scan|generate|show} study.yml [section.key=value ...]   (generate: see aoeb.scripts.generate)"""
import argparse
import os

from ..config import Config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cmd', choices=['scan', 'generate', 'show'])
    p.add_argument('config', help='YAML file (or "-" for the defaults)')
    p.add_argument('overrides', nargs='*', help='section.key=value')
    a = p.parse_args(argv)
    cfg = Config.load(None if a.config == '-' else a.config, a.overrides)
    if a.cmd == 'show':
        import yaml
        print(yaml.safe_dump(cfg.d, sort_keys=False))
    elif a.cmd == 'generate':
        from .generate import main as generate
        generate([a.config] * (a.config != '-') + a.overrides)
    else:
        prefix = cfg.d['output']['prefix']
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        modes, totals = cfg.scan()
        totals.to_csv(prefix + '_totals.csv', index=False)
        if len(modes):
            modes.to_csv(prefix + '_modes.csv', index=False)
        cfg.dump(prefix + '_config.yml')
        print(f'{len(totals)} configurations -> {prefix}_totals.csv')


if __name__ == '__main__':
    main()

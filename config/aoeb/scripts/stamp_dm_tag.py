"""Write DM_TAG in the IMs and reconstructors that do not have it (the budget machine refuses them otherwise). Only for files whose DM is KNOWN: check
first that a fresh IM with that DM reproduces the stored one.
    python stamp_dm_tag.py <root> <dm_tag> [--match wvl850] [--match ...] [--exclude 10x10_wvl750] [--force] [--dry-run]
--match: file names must contain one of these; --force: replace an existing tag."""
import argparse
import glob
import os.path as op

from astropy.io import fits

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('root')
ap.add_argument('dm_tag')
ap.add_argument('--match', action='append', default=[])
ap.add_argument('--exclude', action='append', default=[])
ap.add_argument('--force', action='store_true')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
done = kept = 0
for f in sorted(glob.glob(op.join(a.root, 'im', '*.fits')) + glob.glob(op.join(a.root, 'rec', '*.fits'))):
    name = op.basename(f)
    if (a.match and not any(m in name for m in a.match)) or any(x in name for x in a.exclude):
        continue
    if not a.force and fits.getheader(f).get('DM_TAG') is not None:
        kept += 1
        continue
    if not a.dry_run:
        with fits.open(f, mode='update') as h:
            h[0].header['DM_TAG'] = a.dm_tag
    done += 1
print(f'{"would tag" if a.dry_run else "tagged"} {done} files with {a.dm_tag}; {kept} already tagged')

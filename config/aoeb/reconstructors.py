"""Reconstructors: SVD pseudo-inverse, maximum likelihood and MAP (MMSE).

D is the interaction matrix (n_slopes, N) in slopes per command unit; modes are in command units.
  SVD : R = D^+                                  (what the SPECULA calibration uses)
  ML  : R = (D^T Cn^-1 D)^-1 D^T Cn^-1           (weighted least squares)
  MAP : R = (D^T Cn^-1 D + Ca^-1)^-1 D^T Cn^-1   (= Ca D^T (D Ca D^T + Cn)^-1)
Cn: slope noise covariance (diagonal), Ca: modal prior covariance (diagonal, command^2).
ML and MAP depend on the flux through Cn: the file tag carries the photons per frame.
"""
import os

import numpy as np
from astropy.io import fits
from scipy.linalg import cho_factor, cho_solve



def svd_reconstructor(D, rcond=0.0):
    u, s, vt = np.linalg.svd(np.asarray(D, float), full_matrices=False)
    keep = s > rcond * s.max()
    return (vt.T[:, keep] / s[keep]) @ u[:, keep].T


def weighted_reconstructor(D, cn_diag, ca_diag=None):
    """ML (ca_diag None) or MAP reconstructor: (D^T Cn^-1 D [+ Ca^-1])^-1 D^T Cn^-1."""
    D = np.asarray(D, float)
    wd = D / np.asarray(cn_diag, float)[:, None]
    h = D.T @ wd
    if ca_diag is not None:
        h = h + np.diag(1.0 / np.asarray(ca_diag, float))
    return cho_solve(cho_factor(h), wd.T)


def compute_reconstructor(im, N, method='svd', cn_diag=None, ca_diag=None):
    """Reconstructor for the first N modes of `im` (n_slopes, n_modes_total); method 'svd', 'ml' or 'map'."""
    D = np.asarray(im, float)[:, :N]
    if method == 'svd':
        return svd_reconstructor(D)
    if method not in ('ml', 'map') or cn_diag is None or (method == 'map' and ca_diag is None):
        raise ValueError(f"method '{method}' needs cn_diag (and ca_diag for 'map')")
    return weighted_reconstructor(D, cn_diag, ca_diag[:N] if method == 'map' else None)


def save_rec(path, rec, overwrite=True, **info):
    """Reconstructor in the SPECULA format (empty primary HDU + 'REC' image); `info` (method, n_phot, ...) goes into the header."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['PUP_TAG'] = ''
    hdr['SA_TAG'] = ''
    hdr['NORMFACT'] = 0.0
    for k, v in info.items():
        hdr[k.upper()[:8]] = v if v is not None else 'none'
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr), fits.ImageHDU(data=np.asarray(rec), name='REC')])
    hdul.writeto(path, overwrite=overwrite)
    hdul.close()


def read_rec(path):
    return np.asarray(fits.getdata(path))

"""Unmodulated pyramid WFS: diffractive (nonlinear) model, exact linearisation
around a flat wavefront, and its adjoint (for the analytic covariance model)."""
import numpy as np


class Pyramid:
    def __init__(self, n_pup, os, pad=2.5, sep=0.6):
        """n_pup: simulation pixels across the pupil; os: simulation pixels per
        detector pixel (n_pup/os detector pixels across each pupil image)."""
        self.n, self.os = n_pup, os
        N = int(np.ceil(pad * n_pup / 2) * 2)
        self.N = N
        self.shift = int(round(sep * n_pup))
        c = (np.arange(n_pup) - (n_pup - 1) / 2) / (n_pup / 2)
        X, Y = np.meshgrid(c, c)
        self.P = (X ** 2 + Y ** 2 <= 1.0)
        u = np.fft.fftfreq(N) * N
        U, V = np.meshgrid(u, u)
        self.mask = np.exp(2j * np.pi * self.shift * (np.abs(U) + np.abs(V)) / N)
        self.o0 = N // 2 - n_pup // 2  # pupil embedding offset (centred)
        # four pupil-image windows (in the N x N detector-plane array)
        c0 = N // 2 - n_pup // 2
        self.win = [((c0 + sy) % N, (c0 + sx) % N) for sy in (-self.shift, self.shift)
                    for sx in (-self.shift, self.shift)]
        nd = n_pup // os
        self.nd = nd
        fill = self.P.reshape(nd, os, nd, os).mean((1, 3))
        self.valid = fill > 0.5
        E0 = self._embed(self.P.astype(complex))
        self.A0 = self._F(E0)  # flat-wavefront field in detector plane
        I0 = np.abs(self.A0) ** 2
        self.I0 = I0
        self.norm = self._bin(I0).sum() / self.valid.sum() / 4  # mean flux per valid pixel
        self.s_ref = self._bin(I0) / self.norm
        self.flux_frac_valid = self._bin(I0).sum() / I0.sum()

    # -- optics
    def _embed(self, f):
        g = np.zeros((self.N, self.N), complex)
        g[self.o0:self.o0 + self.n, self.o0:self.o0 + self.n] = f
        return g

    def _crop(self, g):
        return g[self.o0:self.o0 + self.n, self.o0:self.o0 + self.n]

    def _F(self, E):
        return np.fft.ifft2(self.mask * np.fft.fft2(E))

    def _FH(self, A):
        return np.fft.ifft2(np.conj(self.mask) * np.fft.fft2(A))

    def _bin(self, I):
        """detector plane -> 4 x n_valid signal vector (unnormalised)."""
        n, os, nd = self.n, self.os, self.nd
        out = []
        for (y, x) in self.win:
            sub = np.roll(np.roll(I, -y, 0), -x, 1)[:n, :n]
            b = sub.reshape(nd, os, nd, os).sum((1, 3))
            out.append(b[self.valid])
        return np.concatenate(out)

    def _bin_T(self, s):
        """adjoint of _bin: signal vector -> detector-plane weights."""
        n, os, nd = self.n, self.os, self.nd
        nv = self.valid.sum()
        W = np.zeros((self.N, self.N))
        for q, (y, x) in enumerate(self.win):
            b = np.zeros((nd, nd))
            b[self.valid] = s[q * nv:(q + 1) * nv]
            up = np.repeat(np.repeat(b, os, 0), os, 1)
            tmp = np.zeros((self.N, self.N))
            tmp[:n, :n] = up
            W += np.roll(np.roll(tmp, y, 0), x, 1)
        return W

    # -- sensor
    def signal(self, phi):
        """nonlinear normalised signal minus reference; phi in rad on the pupil grid."""
        E = self._embed(self.P * np.exp(1j * phi))
        I = np.abs(self._F(E)) ** 2
        return self._bin(I) / self.norm - self.s_ref

    def signal_lin(self, phi):
        dA = self._F(self._embed(1j * self.P * phi))
        dI = 2 * np.real(np.conj(self.A0) * dA)
        return self._bin(dI) / self.norm

    def signal_lin_T(self, s):
        """adjoint of signal_lin: signal-space vector -> phase-space weights (pupil grid)."""
        W = self._bin_T(s) / self.norm
        g = self._FH(W * self.A0)
        return self.P * 2 * np.imag(self._crop(g)) * -1 * -1  # see dot test

    def noise_var(self, n_ph_pix):
        """Photon-noise variance of each normalised signal for n_ph_pix photons
        per valid pixel on average (flat wavefront)."""
        Ib = self._bin(self.I0) / self.norm  # mean 1 per valid pixel
        return Ib / n_ph_pix


def dot_test(py, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.standard_normal((py.n, py.n)) * py.P
    s = rng.standard_normal(4 * py.valid.sum())
    a = s @ py.signal_lin(phi)
    b = np.sum(phi * py.signal_lin_T(s))
    return a, b

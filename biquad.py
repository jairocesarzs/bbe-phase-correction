"""Filtros biquad IIR (receitas RBJ "Audio EQ Cookbook") em cascata de seções de 2ª ordem (SOS), com estado preservado
entre blocos. Processamento em float64. Requer numpy e scipy."""
import math

import numpy as np
from scipy import signal

Q_BUTTER = 1.0 / math.sqrt(2.0)


def _norm(b0, b1, b2, a0, a1, a2):
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0], dtype=np.float64)


def _w0(fs, f0):
    f0 = min(max(float(f0), 1.0), fs * 0.499)
    return 2.0 * math.pi * f0 / fs


def peaking(fs, f0, gain_db, q):
    A = 10 ** (gain_db / 40.0)
    w0 = _w0(fs, f0)
    al = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    return _norm(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A)


def low_shelf(fs, f0, gain_db, slope=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = _w0(fs, f0)
    c, s = math.cos(w0), math.sin(w0)
    al = s / 2.0 * math.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    sq = 2 * math.sqrt(A) * al
    return _norm(A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq),
                 (A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq)


def high_shelf(fs, f0, gain_db, slope=1.0):
    A = 10 ** (gain_db / 40.0)
    w0 = _w0(fs, f0)
    c, s = math.cos(w0), math.sin(w0)
    al = s / 2.0 * math.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
    sq = 2 * math.sqrt(A) * al
    return _norm(A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq),
                 (A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq)


def lowpass(fs, f0, q=Q_BUTTER):
    w0 = _w0(fs, f0)
    c, al = math.cos(w0), math.sin(w0) / (2 * q)
    return _norm((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al)


def highpass(fs, f0, q=Q_BUTTER):
    w0 = _w0(fs, f0)
    c, al = math.cos(w0), math.sin(w0) / (2 * q)
    return _norm((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al)


def allpass(fs, f0, q=Q_BUTTER):
    w0 = _w0(fs, f0)
    c, al = math.cos(w0), math.sin(w0) / (2 * q)
    return _norm(1 - al, -2 * c, 1 + al, 1 + al, -2 * c, 1 - al)


class SosFilter:
    """Cascata de biquads com estado por canal. process(x[n, ch]) → y[n, ch] (float64)."""

    def __init__(self, sos=None, channels=2):
        self.channels = channels
        self.sos = np.zeros((0, 6))
        self.zi = None
        if sos is not None and len(sos):
            self.set_sos(sos)

    def set_sos(self, sos, keep_state=True):
        sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
        old_n = len(self.sos)
        self.sos = sos
        if self.zi is None or not keep_state or len(sos) != old_n:
            self.reset()

    def reset(self):
        self.zi = np.zeros((len(self.sos), 2, self.channels)) if len(self.sos) else None

    def process(self, x):
        if not len(self.sos):
            return x
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y

    def response(self, fs, freqs):
        """Resposta em magnitude (dB) nas frequências dadas."""
        if not len(self.sos):
            return np.zeros(len(freqs))
        w = 2 * np.pi * np.asarray(freqs) / fs
        _w, h = signal.sosfreqz(self.sos, worN=w)
        return 20 * np.log10(np.maximum(np.abs(h), 1e-12))


def cascade(*sections):
    return np.vstack([np.atleast_2d(s) for s in sections]) if sections else np.zeros((0, 6))

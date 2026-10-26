"""BBE Sound / BMX — correção de fase: rede de filtros PASSA-TUDO que atrasa os graves (e, menos, os médios) em
milissegundos RELATIVOS aos agudos, alinhando no tempo o impacto dos graves com o ataque dos agudos.

Por que passa-tudo e não "atrasar bandas separadas": atrasar saídas de um crossover cria cancelamentos (vales de até
10 dB) perto das frequências de corte. Uma rede passa-tudo tem magnitude exatamente plana e atraso de grupo
dependente da frequência (maior nos graves). As frequências dos passa-tudo são RESOLVIDAS numericamente para que o atraso
de grupo medido seja o configurado: `low_ms` em 80 Hz e `mid_ms` em 600 Hz, ambos em relação a 6 kHz.
Extras opcionais: "definição" (ar nos agudos, shelf) e reforço de graves."""
import numpy as np
from scipy import optimize, signal

from .biquad import SosFilter, allpass, cascade, high_shelf, low_shelf

REF_LOW, REF_MID, REF_HIGH = 80.0, 600.0, 6000.0
Q_AP = 0.707


def group_delay_ms(sos, fs, f):
    """Atraso de grupo (ms) de uma cascata SOS na frequência f (derivada numérica da fase)."""
    f = np.array([f * 0.97, f * 1.03])
    w = 2 * np.pi * f / fs
    ph = np.unwrap(np.angle(signal.sosfreqz(sos, worN=w)[1]))
    return float(-(ph[1] - ph[0]) / (w[1] - w[0]) / fs * 1000.0)


def relative_delay(sos, fs, f):
    """Atraso em f menos o atraso em 6 kHz (relativo aos agudos)."""
    return group_delay_ms(sos, fs, f) - group_delay_ms(sos, fs, REF_HIGH)


Q_CHOICES = (0.707, 1.0, 1.5, 2.2, 3.2)


def _solve_section(fs, target_ms, ref_f, base_sos=None):
    """(f0, Q) do passa-tudo que dá o atraso relativo desejado em ref_f (somado às seções já existentes).
    O atraso de grupo de um passa-tudo de 2ª ordem tem pico perto de f0 (≈ 4Q/ω0) e cai para cima de f0: por isso
    a busca é feita só em f0 ≥ ref_f (ramo monótono) e o Q sobe até o alvo ser alcançável."""
    hi = fs * 0.40
    best = None
    for q in Q_CHOICES:
        def err(f0, q=q):
            secs = [allpass(fs, f0, q)] + ([base_sos] if base_sos is not None else [])
            return relative_delay(cascade(*secs), fs, ref_f) - target_ms
        if err(ref_f) < 0:                         # com este Q nem o pico alcança o alvo
            best = (ref_f, q)
            continue
        if err(hi) >= 0:
            return (hi, q)
        return (optimize.brentq(err, ref_f, hi, xtol=1e-3), q)
    return best                                    # alvo inalcançável: melhor esforço (pico do maior Q)


class BBE:
    def __init__(self, fs, channels=2):
        self.fs, self.ch = fs, channels
        self.enabled = False
        self.low_ms, self.mid_ms = 2.5, 0.8
        self.definition = 0.0           # 0..1  → até +6 dB de ar nos agudos
        self.bass = 0.0                 # 0..1  → até +5 dB de graves
        self.achieved = (0.0, 0.0)      # atrasos medidos (graves, médios) em ms
        self.ap = SosFilter(channels=channels)
        self._design()
        self._shelves()

    def _design(self):
        """Resolve as duas seções passa-tudo juntas (cada uma contribui um pouco no ponto de referência da outra),
        refinando em algumas passadas."""
        fs = self.fs
        low = mid = None
        sos_low = sos_mid = None
        for _ in range(4):
            if self.mid_ms > 0.02:
                mid = _solve_section(fs, self.mid_ms, REF_MID, sos_low)
                sos_mid = allpass(fs, mid[0], mid[1]) if mid else None
            if self.low_ms > 0.02:
                low = _solve_section(fs, self.low_ms, REF_LOW, sos_mid)
                sos_low = allpass(fs, low[0], low[1]) if low else None
        secs = [x for x in (sos_mid, sos_low) if x is not None]
        sos = cascade(*secs) if secs else np.zeros((0, 6))
        self.ap.set_sos(sos, keep_state=False)
        self.achieved = (relative_delay(sos, fs, REF_LOW), relative_delay(sos, fs, REF_MID)) if len(sos) else (0.0, 0.0)

    def _shelves(self):
        self.def_filter = SosFilter(high_shelf(self.fs, 7000, 6.0 * self.definition, 0.8), self.ch)
        self.bass_filter = SosFilter(low_shelf(self.fs, 90, 5.0 * self.bass, 0.8), self.ch)

    def configure(self, enabled=None, low_ms=None, mid_ms=None, definition=None, bass=None):
        redo = False
        if enabled is not None:
            self.enabled = bool(enabled)
        if low_ms is not None and float(low_ms) != self.low_ms:
            self.low_ms, redo = float(low_ms), True
        if mid_ms is not None and float(mid_ms) != self.mid_ms:
            self.mid_ms, redo = float(mid_ms), True
        if redo:
            self._design()
        if definition is not None or bass is not None:
            self.definition = self.definition if definition is None else float(definition)
            self.bass = self.bass if bass is None else float(bass)
            self._shelves()

    @property
    def active(self):
        return self.enabled

    def process(self, x):
        if not self.enabled:
            return x
        y = self.ap.process(x)
        if self.definition > 0:
            y = self.def_filter.process(y)
        if self.bass > 0:
            y = self.bass_filter.process(y)
        return y

    def response(self, freqs):
        """Magnitude (dB): a rede passa-tudo é plana; só definição e graves mudam a curva."""
        r = 0 * np.asarray(freqs, dtype=float)
        if self.enabled and self.definition > 0:
            r = r + self.def_filter.response(self.fs, freqs)
        if self.enabled and self.bass > 0:
            r = r + self.bass_filter.response(self.fs, freqs)
        return r

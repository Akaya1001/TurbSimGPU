#!/usr/bin/env python3
"""Pruefstand der zertifizierten No-op-Regel (Plan 14.10.2026, Stufe A1).

Rechnet fuer echte TurbSim-Geometrien (Punktreihenfolge wie CreateGrid, Kohaerenz ueber die
echten glibc/libmvec-Tabellen aus ts_kern.c wie im Produktionsbau) die Cholesky zweimal:
  * ts_chol_spec (Produktionskern, mit Zeilenprofil) und
  * ts_chol_skipref (skalare Referenz mit Skip-Regel aus ts_skip.h, jeder uebersprungene Schritt
    wird zusaetzlich mit der echten fma nachgerechnet)
und verlangt: 0 Verletzungen, 0 Wertabweichungen in L, gleiches INFO. Nebenbei entstehen die
Zaehler (Spezifikation, ausgefuehrt, offene Chunks), aus denen der Ertrag ueber die Frequenz
integriert wird.

Schreibt nichts ausser auf stdout. Aufruf:
    python3 turbsim/skip_pruefung.py [--schnell] [--boxen std,w960] [--u 5,11,25]
Exit 0 nur ohne jede Abweichung.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

KERN = Path(__file__).resolve().parent / "kern"
P = ctypes.POINTER
I32 = ctypes.c_int32
DP = P(ctypes.c_double)

FLAGS = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-fno-builtin-exp", "-fno-builtin-sqrt", "-fPIC"]
QUELLEN = (("ts_kern.c", ["-fno-tree-vectorize"]), ("ts_chol_scalar.c", []),
           ("ts_chol_avx2.c", ["-mavx2", "-mfma"]),
           ("ts_chol_avx512.c", ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma"]),
           ("ts_chol_skipref.c", ["-mfma", "-fopenmp"]))


def baue_lib(ziel: Path) -> Path:
    """Kern + Skip-Referenz in eine gemeinsame Bibliothek bauen (Sekunden)."""
    ziel.mkdir(parents=True, exist_ok=True)
    objs = []
    for src, extra in QUELLEN:
        o = ziel / (src[:-2] + ".o")
        subprocess.run(["gcc", *FLAGS, *extra, "-c", str(KERN / src), "-o", str(o)], check=True)
        objs.append(str(o))
    so = ziel / "libts_kern_skip.so"
    subprocess.run(["gcc", "-shared", "-fopenmp", "-o", str(so), *objs, "-lmvec", "-lm"], check=True)
    return so


def lade(so: Path):
    lib = ctypes.CDLL(str(so))
    lib.ts_skip_e.restype = I32; lib.ts_skip_e.argtypes = [ctypes.c_double]
    lib.ts_skip_es.restype = I32; lib.ts_skip_es.argtypes = [ctypes.c_double]
    lib.ts_skip_regel.restype = I32; lib.ts_skip_regel.argtypes = [ctypes.c_double] * 3
    lib.ts_chol_isa_name.restype = ctypes.c_char_p
    lib.ts_skipref_schwelle_setzen.argtypes = [I32]
    lib.ts_chol_skip_setzen.argtypes = [I32]
    return lib


# --- Geometrie wie TurbSim v3.5.3 CreateGrid -------------------------------------------------

def gitter(NY, NZ, W, H, HubHt, usr=None, nabe_extra=False):
    """Punkte in TurbSim-Reihenfolge: User-Punkte zuerst (Duplikate eines Gitterpunkts fallen im
    Gitter weg), dann das Gitter (y schnell, z unten -> oben), dann ggf. der Nabenpunkt."""
    gy = W / (NY - 1); gz = H / (NZ - 1)
    zb = HubHt + 0.5 * min(W, H) - gz * (NZ - 1)
    Y, Z = [], []
    if usr is not None:
        Y.append(usr[0]); Z.append(usr[1])
    for iz in range(NZ):
        for iy in range(NY):
            y = -0.5 * W + gy * iy; z = zb + gz * iz
            if usr is not None and y == usr[0] and z == usr[1]:
                continue
            Y.append(y); Z.append(z)
    if nabe_extra:
        Y.append(0.0); Z.append(HubHt)
    return np.array(Y), np.array(Z)


# IECKAI-Vorgabe (CalcIECScalingParams): InCohB = 0.12/(8.1*Lambda), Lambda = 42 m bei HubHt >= 60 m.
INCOHB_IEC = 0.12 / (8.1 * 42.0)
# Farm-High: UHub aus dem Potenzprofil wie TurbSim (Profiles.f90: U_Ref*(Ht/z_Ref)**PLExp), URef 10 bei 150 m.
U_FARM = 10.0 * (156.0 / 150.0) ** 0.14

BOXEN = {
    # name: (Geometrie, U der Kampagne, dt, NumFreq der Vollbox bei diesem U, InCohB)
    "std":      (lambda: gitter(35, 35, 270.0, 270.0, 137.5), 11.0, 0.05, 43470, INCOHB_IEC),
    "w340":     (lambda: gitter(44, 35, 340.0, 270.0, 137.5, nabe_extra=True), 11.0, 0.05, 43520, INCOHB_IEC),
    "w960":     (lambda: gitter(124, 35, 960.0, 270.0, 137.5, nabe_extra=True), 11.0, 0.05, 44100, INCOHB_IEC),
    "farm3600": (lambda: gitter(59, 61, 290.0, 300.0, 156.0, usr=(12.251, 141.0)), U_FARM, 0.42, 5016, 0.00035273),
    "farm3599": (lambda: gitter(59, 61, 290.0, 300.0, 156.0, usr=(10.0, 141.0)), U_FARM, 0.42, 5016, 0.00035273),
}


def gepackt_dist(Y, Z):
    """Dist in der Reihenfolge der gepackten 'L'-Spalten, Maschinenfolge wie Fortran (ohne FMA)."""
    n = len(Y)
    I, J = np.tril_indices(n)
    o = np.lexsort((I, J)); I, J = I[o], J[o]
    dy = Y[I] - Y[J]; dz = Z[I] - Z[J]
    return n, np.sqrt(dy * dy + dz * dz)


class Kohaerenz:
    """Klassen einmal, je Frequenz Tabelle + Gather — exakt die Produktionsbausteine."""

    def __init__(self, lib, Y, Z, U, incdec=12.0, incohb=INCOHB_IEC):
        self.lib = lib
        self.n, self.dist = gepackt_dist(Y, Z)
        self.distu = self.dist / U
        self.U, self.incdec, self.incohb = U, incdec, incohb
        npk = len(self.dist)
        self.cls = np.zeros(npk, np.int32)
        self.distk = np.zeros(npk); self.distuk = np.zeros(npk); k = I32(0)
        lib.ts_cls_build(ctypes.byref(I32(npk)), ctypes.byref(I32(self.n)), self.dist.ctypes.data_as(DP),
                         self.distu.ctypes.data_as(DP), self.cls.ctypes.data_as(P(I32)),
                         self.distk.ctypes.data_as(DP), self.distuk.ctypes.data_as(DP), ctypes.byref(k))
        self.k = k.value

    def matrix(self, f):
        tab = np.zeros(2 * self.k)
        d = lambda x: ctypes.byref(ctypes.c_double(x))
        self.lib.ts_exp_tab(ctypes.byref(I32(self.k)), self.distk.ctypes.data_as(DP), self.distuk.ctypes.data_as(DP),
                            d(self.U), d(f), d(self.incdec), d(self.incohb), tab.ctypes.data_as(DP))
        ap = np.zeros(len(self.dist)); frow = np.zeros(self.n, np.int32)
        self.lib.ts_coh_fill(ctypes.byref(I32(self.n)), ctypes.byref(I32(1)), self.cls.ctypes.data_as(P(I32)),
                             tab.ctypes.data_as(DP), ap.ctypes.data_as(DP), frow.ctypes.data_as(P(I32)))
        return ap, frow


def chol_spec(lib, ap, n, frow=None):
    a = ap.copy(); info = I32(0)
    fr = frow if frow is not None else np.zeros(n, np.int32)
    lib.ts_chol_spec(ctypes.byref(I32(n)), a.ctypes.data_as(DP), fr.ctypes.data_as(P(I32)),
                     ctypes.byref(I32(1 if frow is not None else 0)), ctypes.byref(info))
    return a, info.value


def chol_skipref(lib, ap, n, skip, kc=8, pruefen=True):
    a = ap.copy(); z = np.zeros(10, np.int64); info = I32(0)
    lib.ts_chol_skipref(ctypes.byref(I32(n)), a.ctypes.data_as(DP), ctypes.byref(I32(skip)), ctypes.byref(I32(kc)),
                        ctypes.byref(I32(1 if pruefen else 0)), z.ctypes.data_as(P(ctypes.c_int64)), ctypes.byref(info))
    return a, info.value, z


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def frequenzen(numfreq, anzahl):
    """IFreq-Stuetzstellen: halb log-, halb linear verteilt, sortiert, eindeutig."""
    lg = np.unique(np.round(np.geomspace(1, numfreq, anzahl // 2)).astype(int))
    li = np.unique(np.round(np.linspace(numfreq / (anzahl - anzahl // 2), numfreq, anzahl - anzahl // 2)).astype(int))
    return np.unique(np.concatenate([lg, li]))


def integriere(f, w, fmax):
    """Mittel ueber f gleichverteilt auf (0, fmax]: stueckweise linear, unterhalb der ersten Stuetzstelle konstant."""
    f = np.concatenate([[0.0], f]); w = np.concatenate([[w[0]], w])
    return float(np.trapezoid(w, f) / fmax) if hasattr(np, "trapezoid") else float(np.trapz(w, f) / fmax)


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--boxen", default="std,w340,w960,farm3600,farm3599")
    ap_.add_argument("--u", default="5,11,25", help="Windgeschwindigkeiten (Farm: 11 -> UHub der Kampagne, 10,055)")
    ap_.add_argument("--anzahl", type=int, default=20, help="Frequenzen je Box und U")
    ap_.add_argument("--kc", type=int, default=8)
    ap_.add_argument("--schnell", action="store_true", help="nur std/farm3600, U=11, 8 Frequenzen")
    ap_.add_argument("--streng", type=int, default=3, help="je Box/U so viele Frequenzen zusaetzlich streng (skip=0, alle Bits) pruefen")
    a = ap_.parse_args(argv)
    boxen = a.boxen.split(","); us = [float(x) for x in a.u.split(",")]; anzahl = a.anzahl
    if a.schnell:
        boxen, us, anzahl = ["std", "farm3600"], [11.0], 8

    with tempfile.TemporaryDirectory(prefix="ts_skip_") as tmp:
        lib = lade(baue_lib(Path(tmp)))
        print(f"Kern-ISA {lib.ts_chol_isa_name().decode()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '-')}, kc={a.kc}")
        fehler = 0
        zusammen = []
        for box in boxen:
            geo, ubasis, dt, numfreq, incohb = BOXEN[box]
            Y, Z = geo()
            for u in us:
                U = ubasis if (abs(u - 11.0) < 1e-12 and box.startswith("farm")) else u
                koh = Kohaerenz(lib, Y, Z, U, incohb=incohb)
                n = koh.n; dicht = n ** 3 / 6.0
                delf = 1.0 / (2 * numfreq * dt); fmax = numfreq * delf
                ifr = frequenzen(numfreq, anzahl)
                streng = set(ifr[np.linspace(0, len(ifr) - 1, min(a.streng, len(ifr))).astype(int)]) if a.streng else set()
                zeilen = []
                t0 = time.time()
                for ifreq in ifr:
                    f = ifreq * delf
                    A, frow = koh.matrix(f)
                    ref, i_ref = chol_spec(lib, A, n, frow)
                    neu, i_neu, z = chol_skipref(lib, A, n, 2, a.kc, True)
                    werte = int(np.count_nonzero(ref != neu)) if i_ref == 0 else 0
                    nullvz = int(np.count_nonzero(bits(ref) != bits(neu))) - werte if i_ref == 0 else 0
                    s_abw = 0
                    if ifreq in streng:                      # Referenz selbst: skip=0 gegen Kern ohne Profil UND ohne Skip, alle Bits
                        lib.ts_chol_skip_setzen(0)           # (mit Skip duerfen nur Nullvorzeichen abweichen)
                        r0, i0 = chol_spec(lib, A, n)
                        lib.ts_chol_skip_setzen(1)
                        n0, j0, _ = chol_skipref(lib, A, n, 0, a.kc, False)
                        s_abw = int(np.count_nonzero(bits(r0) != bits(n0))) + (i0 != j0)
                    bad = int(z[3]) + int(z[8]) + werte + s_abw + (i_ref != i_neu)
                    fehler += bad
                    zeilen.append((f, z[0] / dicht, z[1] / dicht, z[2] / dicht, z[6] / max(z[7], 1), z[3] + z[8], werte, nullvz, s_abw, i_ref, i_neu))
                dauer = time.time() - t0
                fa = np.array([r[0] for r in zeilen])
                w_spec = np.array([r[1] for r in zeilen]); w_exec = np.array([r[2] for r in zeilen]); w_off = np.array([r[3] for r in zeilen])
                m_spec, m_exec, m_off = (integriere(fa, w, fmax) for w in (w_spec, w_exec, w_off))
                print(f"\n== {box} N={n} U={U:g} K={koh.k} f_max={fmax:.3f} Hz ({len(ifr)} Frequenzen, {dauer:.0f} s)")
                print("   f [Hz]   spec   ausgef.  Chunk-offen  frei   Verl. Wertabw. Nullvz. streng INFO")
                for r in zeilen:
                    print(f"  {r[0]:8.4f}  {r[1]:.3f}  {r[2]:.4f}   {r[3]:.4f}     {r[4]:.2f}  {r[5]:5d} {r[6]:7d} {r[7]:7d} {r[8]:6d}  {r[9]}/{r[10]}")
                print(f"  Mittel ueber f: spec {m_spec:.4f}, ausgefuehrt {m_exec:.4f}, Chunk-offen {m_off:.4f} (Anteile von dicht N^3/6)")
                zusammen.append((box, n, U, m_spec, m_exec, m_off))
        print("\n== Zusammenfassung (ueber f gemittelt, Anteil von dicht)")
        print("  Box       N     U      spec    ausgef.  Chunk-offen  Faktor spec/ausgef.")
        for box, n, U, ms, me, mo in zusammen:
            print(f"  {box:8s} {n:5d} {U:6.3f}  {ms:.4f}  {me:.4f}   {mo:.4f}      {ms / me:6.1f}")
        print(f"\nABWEICHUNGEN GESAMT: {fehler}")
        return 0 if fehler == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

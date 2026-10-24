"""Zertifizierte No-op-Regel der dpptrf-Kette (turbsim/kern/ts_skip.h, Stufe A1, Plan 14.10.2026).

Eine FMA fma(-a, b, s) darf nur entfallen, wenn ihr gerundetes Ergebnis bitweise s ist. Die Regel
ts_e(a) + ts_e(b) <= ts_es(s) - 56 muss das beweisbar sichern; diese Tests pruefen sie gegen die echte
Hardware-fma (Grenzfaelle, gezielte Zufallstripel) und auf Kohaerenzmatrizen gegen den
Produktionskern ts_chol_spec: L wertgleich (nur Nullvorzeichen duerfen abweichen), V bitgleich.
"""
import ctypes
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "turbsim"))
sys.path.insert(0, str(ROOT / "tests"))
import skip_pruefung as sp  # noqa: E402
from test_turbsim_kern import _koh  # noqa: E402

P = ctypes.POINTER
DP = P(ctypes.c_double)


@pytest.fixture(scope="module")
def lib_pfad(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc fehlt")
    return sp.baue_lib(tmp_path_factory.mktemp("ts_skip"))


@pytest.fixture(scope="module")
def lib(lib_pfad):
    return sp.lade(lib_pfad)


def d(h):
    """double aus Hex-Gleitkomma-Literal."""
    return float.fromhex(h)


def fma_bits(lib_, a, b, s):
    """Echte fma ueber die Pruefhilfe: liefert (zertifiziert, Verletzung, nur Nullvorzeichen, echter No-op)."""
    z = np.zeros(6, np.int64)
    arr = [np.array([x], np.float64) for x in (a, b, s)]
    lib_.ts_skip_pruef_tripel(ctypes.byref(ctypes.c_int64(1)), *(x.ctypes.data_as(DP) for x in arr),
                              z.ctypes.data_as(P(ctypes.c_int64)))
    return z


# --- Exponentenfelder -------------------------------------------------------------------------

def test_exponentenfelder(lib):
    assert lib.ts_skip_e(0.0) == lib.ts_skip_e(-0.0) == -2048          # exakte Null: eigene Marke
    assert lib.ts_skip_e(5e-324) == lib.ts_skip_e(-d("0x0.fffffffffffffp-1022")) == -1023
    assert lib.ts_skip_e(d("0x1p-1022")) == -1022
    assert lib.ts_skip_e(1.0) == lib.ts_skip_e(-1.9999999) == 0
    assert lib.ts_skip_e(float("inf")) == lib.ts_skip_e(float("nan")) == 4096
    assert lib.ts_skip_es(0.0) == lib.ts_skip_es(5e-324) == -1022
    assert lib.ts_skip_es(d("0x1p-1000")) == -1000
    assert lib.ts_skip_es(float("-inf")) == lib.ts_skip_es(float("nan")) == -8192


# --- Grenzfaelle und Gegenbeispiele -----------------------------------------------------------

@pytest.mark.parametrize("a,b,s", [
    # Schwelle 55 waere falsch: s Zweierpotenz, volle Mantissen -> fma != s
    ("0x1.fffffffffffffp-500", "0x1.fffffffffffffp-555", "0x1p-1000"),
    # reelle Regel fl(a*b)*2^55 <= |s| waere falsch: Produkt rundet in FP64 auf 0
    ("0x1p-500", "0x1p-575", "0x1.0000000000001p-1022"),
])
def test_gegenbeispiele_werden_nicht_uebersprungen(lib, a, b, s):
    a, b, s = d(a), d(b), d(s)
    z = fma_bits(lib, a, b, s)
    assert z[3] == 0                      # wirklich kein No-op ...
    assert lib.ts_skip_regel(a, b, s) == 0  # ... und die Regel ueberspringt nicht


def test_schwelle_genau_56(lib):
    """An der Schwelle (Summe = Es-56) zertifiziert die Regel, eine Stufe darueber nicht."""
    s = d("0x1p-1000")
    a, b = d("0x1.fffffffffffffp-500"), d("0x1.fffffffffffffp-556")
    assert lib.ts_skip_regel(a, b, s) == 1 and fma_bits(lib, a, b, s)[1] == 0
    assert lib.ts_skip_regel(a, d("0x1p-555"), s) == 0


@pytest.mark.parametrize("skip,kc", [(1, 8), (2, 1)])
def test_schwelle_im_choleskypfad(lib, skip, kc):
    """Der Grenzfall der Schwelle 56 im Cholesky-Pfad (Element- und Sketch-Regel): mit Schwelle 55 wuerde
    L21 falsch (0x1p-1000 statt 0x1.fffffffffffffp-1001); die Variante mit b = 2^-556-Klasse muss zertifiziert werden."""
    a, s_ = d("0x1.fffffffffffffp-500"), d("0x1p-1000")
    for b, soll_zert in ((d("0x1.fffffffffffffp-555"), False), (d("0x1.fffffffffffffp-556"), True)):
        ap = np.array([1.0, a, b, 1.0, s_, 1.0])
        ref, i_r = sp.chol_spec(lib, ap, 3)
        neu, i_n, z = sp.chol_skipref(lib, ap, 3, skip, kc, True)
        assert i_r == i_n == 0 and z[3] == 0
        assert np.count_nonzero(sp.bits(ref) != sp.bits(neu)) == 0
        # 4 Spezifikationsschritte; (1,1), (2,2)x2 sind triviale No-ops (1 - winzig), (2,1) ist der Grenzfall
        assert z[0] == 4 and z[0] - z[1] == (4 if soll_zert else 3)
    assert sp.bits(sp.chol_spec(lib, np.array([1.0, a, d("0x1.fffffffffffffp-555"), 1.0, s_, 1.0]), 3)[0])[4] == \
        sp.bits(np.array([d("0x1.fffffffffffffp-1001")]))[0]


def test_inf_nan_nie_uebersprungen(lib):
    inf, nan = float("inf"), float("nan")
    for a, b, s in [(0.0, inf, 1.0), (nan, 0.0, 1.0), (1e-300, 1e-300, inf), (1e-300, 1e-300, nan), (inf, 1e-300, 1.0),
                    (0.0, inf, 2.0 ** 60), (inf, 0.0, 2.0 ** 600), (0.0, nan, 2.0 ** 900)]:
        assert lib.ts_skip_regel(a, b, s) == 0


def test_subnormaler_faktor_gegenbeispiel(lib):
    """Subnormal-Marke -1024 statt -1023 waere falsch: dieses Tripel ist kein No-op."""
    a, b, s = d("0x1.fffffffffffffp+0"), d("0x0.fffffffffffffp-1022"), 2.0 ** -968
    assert fma_bits(lib, a, b, s)[3] == 0 and lib.ts_skip_regel(a, b, s) == 0


def test_pruefinstrumente_zaehlen_verletzungen(lib):
    """Positivkontrolle: mit Pruefschwelle 55 muessen Tripel-Hilfe und Cholesky-Referenz die Verletzung zaehlen."""
    a, b, s = d("0x1.fffffffffffffp-500"), d("0x1.fffffffffffffp-555"), d("0x1p-1000")
    lib.ts_skipref_schwelle_setzen(55)
    try:
        z = fma_bits(lib, a, b, s)
        assert z[0] == 1 and z[1] == 1 and z[5] == 1
        ap = np.array([1.0, a, b, 1.0, s, 1.0])
        for skip, kc in ((1, 8), (2, 1)):
            _, _, zz = sp.chol_skipref(lib, ap, 3, skip, kc, True)
            assert zz[3] >= 1
    finally:
        lib.ts_skipref_schwelle_setzen(56)
    assert fma_bits(lib, a, b, s)[1] == 0


def _tripel(rng, m):
    """Zufallstripel, gehaeuft um die Schwelle: Es zufaellig ueber den ganzen Bereich (auch subnormal,
    Null, Zweierpotenzen), e(a)+e(b) in [Es-60, Es-50], Mantissen zufaellig/voll/leer, Vorzeichen zufaellig."""
    def baue(feld, mant, sign):
        return ((sign.astype(np.uint64) << np.uint64(63)) | (feld.astype(np.uint64) << np.uint64(52)) | mant).view(np.float64)

    def mantissen(k):
        m = rng.integers(0, 1 << 52, size=k, dtype=np.uint64)
        wahl = rng.random(k)
        m[wahl < 0.15] = (1 << 52) - 1        # volle Mantisse
        m[(wahl >= 0.15) & (wahl < 0.35)] = 0  # Zweierpotenz
        return m
    fs = rng.integers(0, 2047, size=m)        # Feld von s (0 = subnormal/Null)
    fs[rng.random(m) < 0.05] = 0
    s = baue(fs, mantissen(m), rng.integers(0, 2, m))
    s[rng.random(m) < 0.01] = 0.0
    es = np.maximum(fs, 1) - 1023
    summe = es - rng.integers(50, 61, size=m)
    ea = rng.integers(-1023, 1024, size=m)                      # -1023: subnormaler Faktor
    ea = np.clip(ea, np.maximum(summe - 1023, -1023), np.minimum(summe + 1023, 1023))
    eb = summe - ea
    gueltig = (eb >= -1023) & (eb <= 1023)
    fa = np.clip(ea + 1023, 0, 2046); fb = np.clip(eb + 1023, 0, 2046)
    a = baue(fa, mantissen(m), rng.integers(0, 2, m))
    b = baue(fb, mantissen(m), rng.integers(0, 2, m))
    a[rng.random(m) < 0.01] = 0.0                               # exakte Null als Faktor
    b[rng.random(m) < 0.01] = -0.0
    return a[gueltig], b[gueltig], s[gueltig]


def test_zufallstripel_gegen_echte_fma(lib):
    rng = np.random.default_rng(14102026)
    tot = np.zeros(6, np.int64)
    for _ in range(10):
        a, b, s = _tripel(rng, 2_000_000)
        lib.ts_skip_pruef_tripel(ctypes.byref(ctypes.c_int64(len(a))), a.ctypes.data_as(DP), b.ctypes.data_as(DP),
                                 s.ctypes.data_as(DP), tot.ctypes.data_as(P(ctypes.c_int64)))
    zert, verl, nullvz, noop, verpasst, uneins = (int(x) for x in tot)
    assert verl == 0
    assert uneins == 0                         # Pruefregel der Referenz == ts_skip der Produktion
    assert zert > 1_000_000                    # die Regel greift im Grenzband tatsaechlich
    assert verpasst < noop                     # konservativ, aber nicht leer


# --- Matrixebene ------------------------------------------------------------------------------

# Mit dem Uebergangsband, in dem der Sketch erstmals Chunks schliesst und r = 1/L_cc >> 1 ist
# (35x35: ~0,05-0,1 Hz, 20x15: ~0,15-0,4 Hz) — dort fallen unsichere Varianten auf (Mutationstest 14.10.).
FAELLE = [(10, 6, 14.1, f) for f in (0.02, 0.5, 3.0, 8.0)] + [(20, 15, 7.06, f) for f in (0.1, 0.15, 0.22, 0.38, 1.0, 5.0)] \
       + [(35, 35, 270 / 34, f) for f in (0.05, 0.062, 0.075, 0.09, 0.3, 3.0, 10.0)]


@pytest.mark.parametrize("NY,NZ,dd,f", FAELLE)
def test_referenz_ohne_skip_ist_bitgleich_zum_kern(lib, NY, NZ, dd, f):
    ap, n, _ = _koh(NY, NZ, dd, f)
    lib.ts_chol_skip_setzen(0)                 # Produktionskern ohne Skip: streng bitgleich inkl. Nullvorzeichen
    try:
        ref, i_r = sp.chol_spec(lib, ap, n)
    finally:
        lib.ts_chol_skip_setzen(1)
    neu, i_n, z = sp.chol_skipref(lib, ap, n, 0, 8, False)
    assert i_r == i_n == 0
    assert np.count_nonzero(sp.bits(ref) != sp.bits(neu)) == 0
    assert z[0] == z[1]                        # ohne Skip wird jeder Spezifikationsschritt ausgefuehrt


@pytest.mark.parametrize("NY,NZ,dd,f", FAELLE)
@pytest.mark.parametrize("skip,kc", [(1, 8), (2, 4), (2, 8), (2, 16)])
def test_skip_wertgleich_ohne_verletzung(lib, NY, NZ, dd, f, skip, kc):
    ap, n, frow = _koh(NY, NZ, dd, f)
    ref, i_r = sp.chol_spec(lib, ap, n, frow)
    neu, i_n, z = sp.chol_skipref(lib, ap, n, skip, kc, True)
    assert i_r == i_n == 0
    assert z[3] == 0                           # kein uebersprungener Schritt haette s veraendert
    assert z[8] == 0                           # Sketch-Zertifikat impliziert die Elementregel (glueckunabhaengig)
    assert np.count_nonzero(ref != neu) == 0   # Werte gleich; nur Nullvorzeichen duerfen abweichen
    assert z[4] == z[0] - z[1]                 # jeder uebersprungene Schritt wurde nachgerechnet


def _h2c(L, n, sigma, cs):
    """H2Coeffs der Referenz: V_i = sum_{j<=i} (L_ij * F_ij) * cs_j, j aufsteigend, mul und add getrennt, V ab +0."""
    I, J = np.tril_indices(n); o = np.lexsort((I, J)); I, J = I[o], J[o]
    F = np.where(I == J, np.sqrt(np.abs(sigma[J])), np.sqrt(np.sqrt(np.abs(sigma[I] * sigma[J]))))
    H = L * F
    V = np.zeros(n)
    e = 0
    for j in range(n):
        m = n - j
        V[j:] = V[j:] + H[e:e + m] * cs[j]
        e += m
    return V


@pytest.mark.parametrize("NY,NZ,dd,f", [(20, 15, 7.06, 1.0), (35, 35, 270 / 34, 3.0), (35, 35, 270 / 34, 10.0)])
def test_v_bitgleich(lib, NY, NZ, dd, f):
    ap, n, frow = _koh(NY, NZ, dd, f)
    ref, _ = sp.chol_spec(lib, ap, n, frow)
    neu, _, _ = sp.chol_skipref(lib, ap, n, 2, 8, False)
    rng = np.random.default_rng(3)
    sigma = rng.uniform(0.1, 5.0, n); phi = rng.uniform(0, 2 * np.pi, n)
    for cs in (np.cos(phi), np.sin(phi)):
        assert np.count_nonzero(sp.bits(_h2c(ref, n, sigma, cs)) != sp.bits(_h2c(neu, n, sigma, cs))) == 0


def test_info_pfad(lib):
    ap = _koh(20, 15, 7.06, 0.5)[0].copy(); n = 300; k = 37
    ap[int(k * n - k * (k - 1) // 2)] = -5.0
    assert sp.chol_spec(lib, ap, n)[1] == 38
    for skip in (0, 1, 2):
        assert sp.chol_skipref(lib, ap, n, skip, 8, True)[1] == 38
    sing = np.array([1.0, 1.0, 1.0])            # A = [[1,1],[1,1]]: Diagonale wird exakt 0 -> INFO 2
    assert sp.chol_spec(lib, sing, 2)[1] == 2
    for skip in (0, 1, 2):
        assert sp.chol_skipref(lib, sing, 2, skip, 1, True)[1] == 2


def test_ertrag_bei_hoher_frequenz(lib):
    """Schutz gegen eine Regel, die nie greift: std-Gitter bei 3 Hz laesst < 5 % der Spezifikationsschritte uebrig."""
    ap, n, _ = _koh(35, 35, 270 / 34, 3.0)
    _, _, z = sp.chol_skipref(lib, ap, n, 1, 8, False)
    assert z[1] < 0.05 * z[0]
    _, _, z2 = sp.chol_skipref(lib, ap, n, 2, 8, False)
    assert z2[2] < 0.2 * z2[0]                 # der Sketch schliesst die meisten Chunks (gemessen ~0,11)
    assert z2[1] == z[1]                       # zweistufig fuehrt genau die Schritte der Elementregel aus


def test_produktionsgeometrie_und_kohaerenz(lib):
    """Pruefstand-Bausteine: Punktzahlen wie TurbSim (Nabe/User-Punkt) und A wie im Produktionsbau."""
    assert len(sp.BOXEN["std"][0]()[0]) == 1225
    assert len(sp.BOXEN["w340"][0]()[0]) == 1541
    assert len(sp.BOXEN["w960"][0]()[0]) == 4341
    assert len(sp.BOXEN["farm3600"][0]()[0]) == 3600
    assert len(sp.BOXEN["farm3599"][0]()[0]) == 3599
    Y, Z = sp.BOXEN["std"][0]()
    koh = sp.Kohaerenz(lib, Y, Z, 11.0)
    assert koh.k == 1240
    A, frow = koh.matrix(1.0)
    assert np.all(A[np.cumsum(np.r_[0, np.arange(1225, 1, -1)])] == 1.0)   # Diagonale exakt 1


# --- Produktionskern: konstruierte Grenzfaelle (Mutationstest 14.10.) -------------------------------
# Kohaerenzmatrizen treffen nie das eine Binaden-Band, in dem sich Schwelle 55 von 56 oder ein veraltetes
# von einem frischen Exponentenminimum unterscheidet. Diese Matrizen tun es gezielt: Kachel (Gruppenskizze),
# Schatten esh zwischen zwei Nachlaeufen, Kachelminimum zwischen zwei Chunks, Dreieckskopf (Block- und
# Elementregel), Randkachel bei n % 8 == 1 und exakt skalierte Kohaerenzmatrizen (|L| >> 1 bzw. << 1).
# Massstab: Produktionskern mit Skip gegen denselben Kern ohne Skip (mit und ohne Zeilenprofil), Werte + INFO.
_GEZIELT = r"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[3])
import skip_pruefung as sp
from test_turbsim_kern import _koh
lib = sp.lade(Path(sys.argv[1])); isa = lib.ts_chol_isa_name().decode()
h = float.fromhex

def gepackt(A):
    n = A.shape[0]; I, J = np.tril_indices(n); o = np.lexsort((I, J))
    return A[I[o], J[o]].copy(), I[o], J[o]

def frow_von(ap, n, I, J):
    fr = np.zeros(n, np.int32)
    for e in np.nonzero(ap != 0.0)[0]:
        if fr[I[e]] == 0: fr[I[e]] = J[e] + 1
    return fr

def abw(ap, n, fr=None):
    lib.ts_chol_skip_setzen(0); ref, i0 = sp.chol_spec(lib, ap, n); lib.ts_chol_skip_setzen(1)
    a1, i1 = sp.chol_spec(lib, ap, n)
    d = int(np.count_nonzero(ref != a1)) + (i0 != i1)
    if fr is not None:
        a2, i2 = sp.chol_spec(lib, ap, n, fr); d += int(np.count_nonzero(ref != a2)) + (i0 != i2)
    return d, ref

tot = 0
# 1) Schwelle 56 in Kachel und Kopf, n = 64
n = 64; A = np.eye(n)
A[40, 0] = h("0x1.fffffffffffffp-500"); A[32, 0] = h("0x1.fffffffffffffp-555"); A[40:64, 32:40] = 2.0 ** -1000
A[48, 1] = A[49, 1] = h("0x1.fffffffffffffp-528")
for i in range(48, 56):
    for c in range(48, i): A[i, c] = 2.0 ** -1001
ap, I, J = gepackt(A); d, ref = abw(ap, n, frow_von(ap, n, I, J)); tot += d
L = np.zeros((n, n)); L[I, J] = ref
assert L[40, 32] == h("0x1.fffffffffffffp-1001") and L[49, 48] != 2.0 ** -1001   # Grenzfaelle wirklich scharf

# 2) veraltete Minima, n = 128: s = 2^-1000 wird von L(i,kd)L(c,kd) = 2^-1001 halbiert, danach ein
#    Grenzschritt mit e(a)+e(b) = Es_alt-56 = Es_neu-55, der s wirklich aendert
n = 128
def fall(i, c, kd, kg, block, ga="0x1.fffffffffffffp-500", gb="0x1.fffffffffffffp-556"):
    A = np.eye(n); r0, r1, c0 = block
    for r in range(r0, r1):
        for cc in range(c0, c0 + 8):
            if r > cc: A[r, cc] = 2.0 ** -1000
    A[i, kd] = 2.0 ** -500; A[c, kd] = 2.0 ** -501; A[i, kg] = h(ga); A[c, kg] = h(gb)
    return gepackt(A)
faelle = [(i, c, 60, 64, (104, 128, 96)) for i in (104, 111, 112, 120, 127) for c in (96, 99, 103)]   # Schatten esh
faelle += [(i, c, 48, 56, (104, 128, 96)) for i in (104, 112, 127) for c in (96, 103)]                # Kachelminimum
faelle += [(i, c, 48, 56, (96, 104, 96)) for i, c in ((101, 96), (103, 102), (97, 96))]               # Kopf, Elementregel
for f in faelle:
    ap, I, J = fall(*f); tot += abw(ap, n, frow_von(ap, n, I, J))[0]
for i, c in ((101, 96), (103, 102)):                                                                 # Kopf, Blockzertifikat
    ap, I, J = fall(i, c, 48, 56, (96, 104, 96), "0x1.fffffffffffffp-528", "0x1.fffffffffffffp-528")
    tot += abw(ap, n, frow_von(ap, n, I, J))[0]
# Kopf vertauscht: e(L(c,k)) > e(L(i,k)), damit z[i]+z[i] statt z[i]+z[c] auffaellt
ap, I, J = fall(101, 96, 48, 56, (96, 104, 96), "0x1.fffffffffffffp-560", "0x1.fffffffffffffp-496")
tot += abw(ap, n, frow_von(ap, n, I, J))[0]

# 3) Randkachel bei n % 8 == 1
n = 129; A = np.eye(n); A[128, 0] = 0.5; A[97, 0] = 0.5; A[128, 97] = 0.3
ap, I, J = gepackt(A); tot += abw(ap, n)[0]

# 4) exakt skalierte Kohaerenzmatrizen (Skip-Regel skaleninvariant; |L| >= 2 bzw. winzig)
for NY, NZ, dd, f in ((20, 15, 7.06, 0.22), (35, 35, 270 / 34, 0.062)):
    ap0, n, fr = _koh(NY, NZ, dd, f)
    for sk in (2.0 ** 400, 2.0 ** -600):
        tot += abw(ap0 * sk, n, fr)[0]
print("GEZIELT", isa, tot)
"""


@pytest.mark.parametrize("isa", ["avx512", "avx2", "scalar"])
def test_produktionskern_gezielte_grenzfaelle(isa, lib_pfad):
    """Je ISA in eigenem Prozess (die ISA-Wahl ist je Prozess fest)."""
    import os
    import subprocess
    if isa == "avx512" and "avx512f" not in open("/proc/cpuinfo").read():
        pytest.skip("CPU ohne AVX-512")
    env = dict(os.environ, TS_CHOL_ISA=isa, OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", _GEZIELT, str(lib_pfad), str(ROOT / "turbsim"), str(ROOT / "tests")],
                       capture_output=True, text=True, timeout=600, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert f"GEZIELT {isa} 0" in r.stdout, r.stdout

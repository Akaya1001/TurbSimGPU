"""Bitgleiche Bausteine der TurbSim-Frequenzschleife (turbsim/kern) gegen die
echten Bibliotheken: glibc exp, libmvec _ZGVbN2v_exp, OpenBLAS dpptrf_.

Der Massstab ist Bitgleichheit, keine Toleranz. Die Tests bauen den Kern selbst
(gcc, Sekunden) in ein temporaeres Verzeichnis, damit sie nicht an einem
Bauverzeichnis haengen; ohne gcc oder OpenBLAS werden sie uebersprungen.

Hintergrund (Plan vom 30.09.2026): Die Referenz /usr/local/bin/turbsim (v3.5.3)
rechnet die Kohaerenz mit zwei verschiedenen exp-Varianten (Paare: libmvec,
letztes Element ungerader Spalten: glibc exp) und die Cholesky mit OpenBLAS
dpptrf_, dessen Rechnung je Element eine FMA-Kette in aufsteigendem k ist.
"""
import ctypes
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
KERN = ROOT / "turbsim" / "kern"
OPENBLAS = Path("/usr/lib/x86_64-linux-gnu/libopenblas.so.0")
P = ctypes.POINTER
I32 = ctypes.c_int32


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc fehlt")
    d = tmp_path_factory.mktemp("ts_kern")
    fl = ["-O2", "-ffp-contract=off", "-fno-fast-math", "-fno-builtin-exp", "-fno-builtin-sqrt", "-fPIC"]
    objs = []
    for src, extra in (("ts_kern.c", ["-fno-tree-vectorize"]), ("ts_chol_scalar.c", []),
                       ("ts_chol_avx2.c", ["-mavx2", "-mfma"]),
                       ("ts_chol_avx512.c", ["-mavx512f", "-mavx512dq", "-mavx512vl", "-mfma"])):
        o = d / (src[:-2] + ".o")
        subprocess.run(["gcc", *fl, *extra, "-c", str(KERN / src), "-o", str(o)], check=True)
        objs.append(str(o))
    so = d / "libts_kern.so"
    subprocess.run(["gcc", "-shared", "-o", str(so), *objs, "-lmvec", "-lm"], check=True)
    # Kein FMA ausserhalb der SIMD-Kerne (die Referenz rechnet ohne FMA)
    dis = subprocess.run(["objdump", "-d", objs[0], objs[1]], capture_output=True, text=True).stdout
    assert "vfmadd" not in dis and "vfnmadd" not in dis
    L = ctypes.CDLL(str(so))
    L.ts_test_exp_vec.restype = ctypes.c_double; L.ts_test_exp_vec.argtypes = [ctypes.c_double]
    L.ts_test_exp_scal.restype = ctypes.c_double; L.ts_test_exp_scal.argtypes = [ctypes.c_double]
    L.ts_chol_isa_name.restype = ctypes.c_char_p
    return L


@pytest.fixture(scope="module")
def glibc():
    libm = ctypes.CDLL("libm.so.6"); mvec = ctypes.CDLL("libmvec.so.1")
    libm.exp.restype = ctypes.c_double; libm.exp.argtypes = [ctypes.c_double]
    mvec._ZGVbN2v_exp.restype = ctypes.c_double; mvec._ZGVbN2v_exp.argtypes = [ctypes.c_double]  # Lane 0
    return libm, mvec


def gitter_std(NY=35, NZ=35, W=270.0, H=270.0, HubHt=137.5):
    """Punkte wie TurbSim CreateGrid (IY schnell), ohne Extra-Nabenpunkt (NY ungerade)."""
    gy, gz = W / (NY - 1), H / (NZ - 1); zb = HubHt - 0.5 * min(W, H)
    Y = np.array([-0.5 * W + gy * iy for iz in range(NZ) for iy in range(NY)])
    Z = np.array([zb + gz * iz for iz in range(NZ) for iy in range(NY)])
    return Y, Z


def gepackt(Y, Z):
    n = len(Y); I, J = np.tril_indices(n); o = np.lexsort((I, J)); I, J = I[o], J[o]
    dy, dz = Y[I] - Y[J], Z[I] - Z[J]
    return n, I, J, np.sqrt(dy * dy + dz * dz)


# --- exp-Varianten --------------------------------------------------------------

def test_exp_varianten_sind_die_echten_glibc_funktionen(lib, glibc):
    libm, mvec = glibc
    rng = np.random.default_rng(1)
    xs = np.concatenate([rng.uniform(-800, 0, 50000), rng.uniform(-708.5, -708.3, 5000),
                         [-0.0, -745.1332191019412, -745.1332191019411, -1e-300, -1024.0, -10879.0]])
    bad = 0
    for x in xs:
        x = float(x)
        bad += struct.pack("<d", lib.ts_test_exp_vec(x)) != struct.pack("<d", mvec._ZGVbN2v_exp(x))
        bad += struct.pack("<d", lib.ts_test_exp_scal(x)) != struct.pack("<d", libm.exp(x))
    assert bad == 0


def test_vektor_und_skalar_exp_unterscheiden_sich_wirklich(glibc):
    """Begruendet die Paritaetsregel: ~43 % der Argumente weichen um bis zu 3 ULP ab."""
    libm, mvec = glibc
    rng = np.random.default_rng(2); xs = rng.uniform(-300, -1, 20000)
    anteil = np.mean([libm.exp(float(x)) != mvec._ZGVbN2v_exp(float(x)) for x in xs])
    assert 0.3 < anteil < 0.6


# --- Klassen + Gather ------------------------------------------------------------

def test_gather_ist_bitgleich_zur_elementweisen_referenz(lib, glibc):
    libm, mvec = glibc
    Y, Z = gitter_std(); n, I, J, dist = gepackt(Y, Z); U = 11.0; distu = dist / U
    npk = len(dist)
    cls = np.zeros(npk, np.int32); distk = np.zeros(npk); distuk = np.zeros(npk); k = I32(0)
    lib.ts_cls_build(ctypes.byref(I32(npk)), ctypes.byref(I32(n)), dist.ctypes.data_as(P(ctypes.c_double)),
                     distu.ctypes.data_as(P(ctypes.c_double)), cls.ctypes.data_as(P(I32)),
                     distk.ctypes.data_as(P(ctypes.c_double)), distuk.ctypes.data_as(P(ctypes.c_double)), ctypes.byref(k))
    assert k.value == 1240                      # std-Box: 1240 verschiedene Dist-Bitmuster
    InCDec, InCohB = 12.0, 0.12 / (8.1 * 42.0)
    DelF = 1.0 / (2 * 1470 * 0.05)
    laenge = n - (J + 1) + 1
    skalar = (I == n - 1) & (laenge % 2 == 1)   # letztes Element ungerader Spalten: skalares exp
    assert int(skalar.sum()) == 613
    for ifreq in (1, 700, 1470):
        f = ifreq * DelF
        tab = np.zeros(2 * k.value)
        lib.ts_exp_tab(ctypes.byref(k), distk.ctypes.data_as(P(ctypes.c_double)), distuk.ctypes.data_as(P(ctypes.c_double)),
                       ctypes.byref(ctypes.c_double(U)), ctypes.byref(ctypes.c_double(f)), ctypes.byref(ctypes.c_double(InCDec)),
                       ctypes.byref(ctypes.c_double(InCohB)), tab.ctypes.data_as(P(ctypes.c_double)))
        trh = np.zeros(npk); frow = np.zeros(n, np.int32)
        lib.ts_coh_fill(ctypes.byref(I32(n)), ctypes.byref(I32(1)), cls.ctypes.data_as(P(I32)),
                        tab.ctypes.data_as(P(ctypes.c_double)), trh.ctypes.data_as(P(ctypes.c_double)), frow.ctypes.data_as(P(I32)))
        x = f * distu; y = InCohB * dist; arg = -(InCDec * np.sqrt(x * x + y * y))   # Maschinenfolge der Referenz
        ref = np.array([libm.exp(float(a)) if s else mvec._ZGVbN2v_exp(float(a)) for a, s in zip(arg, skalar)])
        assert np.count_nonzero(_bits(ref) != _bits(trh)) == 0
        assert np.all(trh[I == J] == 1.0)
        # frow: erste Spalte != 0 je Zeile (1-basiert)
        erste = np.full(n, 0, np.int32)
        for e in np.nonzero(trh != 0.0)[0]:
            if erste[I[e]] == 0: erste[I[e]] = J[e] + 1
        assert np.array_equal(erste, frow)


# --- Cholesky ---------------------------------------------------------------------

def _dpptrf(ob, ap, n):
    a = ap.copy(); info = ctypes.c_int(0)
    ob.dpptrf_(ctypes.c_char_p(b"L"), ctypes.byref(ctypes.c_int(n)), a.ctypes.data_as(P(ctypes.c_double)),
               ctypes.byref(info), ctypes.c_size_t(1))
    return a, info.value


def _ts_chol(lib, ap, n, frow=None):
    a = ap.copy(); info = I32(0)
    fr = frow if frow is not None else np.zeros(n, np.int32)
    lib.ts_chol_spec(ctypes.byref(I32(n)), a.ctypes.data_as(P(ctypes.c_double)), fr.ctypes.data_as(P(I32)),
                     ctypes.byref(I32(1 if frow is not None else 0)), ctypes.byref(info))
    return a, info.value


def _koh(NY, NZ, d, f, U=11.0, a=12.0, b=0.00035273):
    Y = np.array([iy * d for iz in range(NZ) for iy in range(NY)]); Z = np.array([iz * d for iz in range(NZ) for iy in range(NY)])
    n, I, J, dist = gepackt(Y, Z); x = f * (dist / U); y = b * dist
    ap = np.exp(-(a * np.sqrt(x * x + y * y)))
    frow = np.zeros(n, np.int32)
    for e in np.nonzero(ap != 0.0)[0]:
        if frow[I[e]] == 0: frow[I[e]] = J[e] + 1
    return ap, n, frow


@pytest.fixture(scope="module")
def openblas():
    if not OPENBLAS.exists():
        pytest.skip("OpenBLAS fehlt")
    ob = ctypes.CDLL(str(OPENBLAS)); ob.openblas_set_num_threads(1)
    return ob


@pytest.mark.parametrize("skip", ["0", "1"])
@pytest.mark.parametrize("isa", ["avx512", "avx2", "scalar"])
def test_cholesky_bitgleich_zu_dpptrf(isa, skip, openblas, tmp_path_factory):
    """Jede ISA-Fassung in einem eigenen Prozess (die ISA-Wahl ist je Prozess fest). Ohne No-op-Elimination
    (TS_CHOL_SKIP=0) streng bitgleich inkl. Nullvorzeichen; mit (Vorgabe) wertgleich — die zertifiziert
    ausgelassenen FMAs koennen nur das Vorzeichen exakter Nullen in L aendern (tests/test_turbsim_skip.py)."""
    if isa == "avx512":
        flags = open("/proc/cpuinfo").read()
        if "avx512f" not in flags:
            pytest.skip("CPU ohne AVX-512")
    prog = f"""
import ctypes, os, sys, numpy as np
sys.path.insert(0, {str(ROOT / 'tests')!r})
os.environ['TS_CHOL_ISA'] = {isa!r}
os.environ['TS_CHOL_SKIP'] = {skip!r}
streng = {skip!r} == '0'
from test_turbsim_kern import _koh, _dpptrf, _ts_chol, _bits, P
lib = ctypes.CDLL(sys.argv[1]); ob = ctypes.CDLL({str(OPENBLAS)!r}); ob.openblas_set_num_threads(1)
lib.ts_chol_isa_name.restype = ctypes.c_char_p
assert lib.ts_chol_isa_name().decode() == {isa!r}
abw = 0
# Dazu das Uebergangsband, in dem die Skizze erstmals Chunks schliesst (20x15: 0,15-0,4 Hz; 35x35: 0,05-0,1 Hz)
faelle = [(g, f) for g in [(10, 6, 14.1), (20, 15, 7.06), (35, 35, 270/34)] for f in (0.02, 0.5, 3.0, 8.0)] \
       + [((20, 15, 7.06), f) for f in (0.15, 0.22, 0.38)] + [((35, 35, 270/34), f) for f in (0.05, 0.062, 0.075, 0.09)]
for (NY, NZ, d), f in faelle:
    ap, n, frow = _koh(NY, NZ, d, f)
    ref, i_r = _dpptrf(ob, ap, n)
    neu, i_n = _ts_chol(lib, ap, n)                 # ohne Zeilenprofil: streng bitgleich inkl. Nullvorzeichen
    neu2, i_2 = _ts_chol(lib, ap, n, frow)          # mit Zeilenprofil: wertgleich
    assert i_r == i_n == i_2 == 0
    abw += int(np.count_nonzero(_bits(ref) != _bits(neu))) if streng else int(np.count_nonzero(ref != neu))
    abw += int(np.count_nonzero(ref != neu2))
rng = np.random.default_rng(7); n = 300; M = rng.standard_normal((n, n)); A = M @ M.T + n*np.eye(n)
I, J = np.tril_indices(n); o = np.lexsort((I, J)); ap = A[I[o], J[o]].copy()
ref, _ = _dpptrf(ob, ap, n); neu, _ = _ts_chol(lib, ap, n); abw += int(np.count_nonzero(_bits(ref) != _bits(neu)))
# INFO-Pfad: negative Diagonale in Spalte 38 (1-basiert)
ap2 = _koh(20, 15, 7.06, 0.5)[0].copy(); k = 37; n2 = 300; ap2[int(k*n2 - k*(k-1)//2)] = -5.0
assert _dpptrf(ob, ap2, n2)[1] == 38 == _ts_chol(lib, ap2, n2)[1]
print('ABW', abw)
"""
    so = None
    for cand in tmp_path_factory.getbasetemp().rglob("libts_kern.so"):
        so = cand
    assert so is not None, "lib-Fixture muss vorher gelaufen sein"
    r = subprocess.run([sys.executable, "-c", prog, str(so)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ABW 0" in r.stdout

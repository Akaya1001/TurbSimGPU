"""Patch-Kette fuer den v3.5.3-Referenzbaum (turbsim/patch_turbsim_*.py):
Fundstrings genau einmal vorhanden, Kette idempotent, Reihenfolge im Bauskript,
Bitgleichheits-Schalter gesetzt. Baut nichts; arbeitet auf einer Kopie der
unveraenderten Referenzquellen: OPENFAST_REF (Vorgabe ~/.local/opt/openfast-3.5.3), ein sauberer Klon von v3.5.3.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TS = ROOT / "turbsim"
REF = Path(os.environ.get("OPENFAST_REF", Path.home() / ".local" / "opt" / "openfast-3.5.3")) / "modules" / "turbsim" / "src"
KETTE_TS = ["instr", "omp353", "gather", "chol", "eye", "farm3", "rest", "h2c", "fehlerpfad", "gpu", "gpunur"]
KETTE_MAIN = ["dump", "zeiten", "plan"]


@pytest.fixture(scope="module")
def quellen(tmp_path_factory):
    if not (REF / "TSsubs.f90").exists():
        pytest.skip("Referenzbaum v3.5.3 fehlt")
    d = tmp_path_factory.mktemp("src")
    for f in ("TSsubs.f90", "TurbSim.f90"):
        shutil.copy(REF / f, d / f)
    return d


@pytest.fixture(scope="module")
def gepatcht(tmp_path_factory):
    """Eigene Kopie, auf der die ganze Kette einmal gelaufen ist (unabhaengig von der Testreihenfolge)."""
    if not (REF / "TSsubs.f90").exists():
        pytest.skip("Referenzbaum v3.5.3 fehlt")
    d = tmp_path_factory.mktemp("gepatcht")
    for f in ("TSsubs.f90", "TurbSim.f90"):
        shutil.copy(REF / f, d / f)
    for name in KETTE_TS:
        _lauf(name, d / "TSsubs.f90")
    for name in KETTE_MAIN:
        _lauf(name, d / "TurbSim.f90")
    return d


def _lauf(name, datei):
    r = subprocess.run([sys.executable, str(TS / f"patch_turbsim_{name}.py"), str(datei)], capture_output=True, text=True)
    assert r.returncode == 0, f"{name}: {r.stderr[-1500:]}"
    return r.stdout


def test_referenzquelle_ist_unveraendert():
    r = subprocess.run(["git", "-C", str(REF.parents[2]), "status", "--short"], capture_output=True, text=True)
    assert r.stdout.strip() == "", f"Referenzbaum {REF.parents[2]} muss sauber bleiben"


def test_kette_laeuft_und_ist_idempotent(quellen):
    ts = quellen / "TSsubs.f90"; main = quellen / "TurbSim.f90"
    for name in KETTE_TS:
        _lauf(name, ts)
    for name in KETTE_MAIN:
        _lauf(name, main)
    stand = ts.read_text(); stand_main = main.read_text()
    for name in KETTE_TS:            # zweiter Durchlauf aendert nichts
        out = _lauf(name, ts)
        assert "weiter" in out
    for name in KETTE_MAIN:
        _lauf(name, main)
    assert ts.read_text() == stand and main.read_text() == stand_main


def test_marker_und_kernaufrufe_vorhanden(gepatcht):
    quellen = gepatcht
    s = (quellen / "TSsubs.f90").read_text()
    for m in ("ts_cls_build", "ts_exp_tab", "ts_coh_fill", "ts_chol_spec", "tsAlleGleich", "tsEyeI", "tsRestFFT", "TRH_t", "tsFehlIFreq", "ts_h2c", "tsAbbIFreq", "tsg_iec_lauf", "#ifdef TURBSIM_GPU"):
        assert m in s, m
    assert "LAPACK_pptrf" not in s.split("SUBROUTINE Coh2H", 1)[1].split("END SUBROUTINE Coh2H", 1)[0]
    assert s.count("EXP( -1.0_ReKi * p%met%InCDec(IVec)") == 0 or "TRH_t(Indx) = EXP(" not in s
    assert "LAPACK_DPPTRF: Leading minor order" in s          # Fehlertext wie NWTC_LAPACK
    assert "TS_DUMP_V" in (quellen / "TurbSim.f90").read_text() and "TS_ZEITEN" in (quellen / "TurbSim.f90").read_text()


def test_nur_gpu_sperre_und_plan_stehen_im_ifdef(gepatcht):
    """TS_GPU=nur und TS_PLAN wirken nur im GPU-Bau: alles zwischen #ifdef TURBSIM_GPU und #endif."""
    import re
    quellen = gepatcht
    s = (quellen / "TSsubs.f90").read_text()
    m = (quellen / "TurbSim.f90").read_text()
    for text, marke, n in ((s, "tsg_pflicht()", 3), (m, "TS_PLAN", 2)):
        bloecke = re.findall(r"#ifdef TURBSIM_GPU\n(.*?)#endif", text, re.S)
        assert sum(b.count(marke) for b in bloecke) == text.count(marke) >= n, marke
    # Sperre greift vor dem CPU-Rueckfall (V nullen) und im Verteiler vor GENERAL/API
    iec = s.split("SUBROUTINE CalcFourierCoeffs_IEC", 1)[1]
    assert iec.index("tsg_pflicht() == 1") < iec.index("ELSE\n         V(:,:,IVec) = 0.0_ReKi")
    vert = s.split("\nSUBROUTINE CalcFourierCoeffs(", 1)[1].split("END SUBROUTINE CalcFourierCoeffs\n", 1)[0]
    assert vert.index("CohMod_GENERAL") < vert.index("CALL CalcFourierCoeffs_General")
    # TS_PLAN endet vor dem ersten grossen Feld (S)
    assert m.index("CALL NormStop()", m.index("TS_PLAN npoints=")) < m.index("CALL AllocAry( S,")
    # nur ein nicht leerer Wert ausser '0' loest TS_PLAN aus; ueberlange Werte (STATUS=-1) zaehlen als gesetzt
    assert "LENGTH=tsPlanLen" in m and "tsPlanStat == -1" in m and "tsPlanAn(1:1) /= '0'" in m


def test_falsche_reihenfolge_scheitert_laut(tmp_path):
    shutil.copy(REF / "TSsubs.f90", tmp_path / "TSsubs.f90")
    r = subprocess.run([sys.executable, str(TS / "patch_turbsim_gather.py"), str(tmp_path / "TSsubs.f90")], capture_output=True, text=True)
    assert r.returncode != 0 and (tmp_path / "TSsubs.f90").read_text() == (REF / "TSsubs.f90").read_text()


def test_bauskript_reihenfolge_und_schalter():
    s = (TS / "build_turbsim_ref353.sh").read_text()
    kette = s[s.index("# --- 1. Patches"):]                    # Kopfkommentar erwaehnt einzelne Patches auch
    pos = [kette.index(f"patch_turbsim_{n}.py") for n in ["instr", "omp353", "dump", "zeiten", "plan", "gather", "chol", "eye", "farm3", "rest", "h2c", "fehlerpfad", "gpu", "gpunur"]]
    assert pos == sorted(pos)
    assert "-ffp-contract=off" in s and "-fno-fast-math" in s and "-lmvec" in s
    assert "DOUBLE_PRECISION=ON" in s and "GIT_DESCRIBE" in s
    assert "v3.5.3" in s                                          # Referenzstand
    assert "ts_chol_scalar.c" in s and "ts_chol_avx512.c" in s and "-mavx512f" in s
    # Kerne ohne SIMD-Flags: kein -march, kein -mfma in den Basis-Flags
    basis = s.split("KERN_FL=")[1].split("\n")[0]
    assert "-mfma" not in basis and "-march" not in basis

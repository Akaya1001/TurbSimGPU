"""Planung des Nur-GPU-Laeufers turbsim/gpu_windfelder.py (17.10.2026): Groessenmodelle gegen gemessene Werte,
Planungslogik, Eingabesuche, Sensorlesen auf einem nachgebauten sysfs. Keine GPU-Last: die Integrationstests rufen das
Binary nur mit TS_PLAN (endet vor jeder Rechnung) und werden ohne Binary uebersprungen.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "turbsim"))
import gpu_windfelder as gw  # noqa: E402

GIB = 1 << 30
BEISPIELE = ROOT / "examples"
BIN = gw.BIN_VORGABE

# TS_PLAN-Zeilen des GPU-Binarys (17.10.2026) fuer Boxen mit gemessenem RSS/VRAM bzw. Bestands-.bts
STD_VOLL = ("TS_PLAN npoints=1225 npacked=750925 numsteps=87040 numfreq=43520 numoutsteps=87000 ngrid=1225 ny=35 nz=35 "
            "ntwr=0 usr=0 scmod=2,0,0 cohgleich=0 wr=100000000 turbmodel=1 desclen=96")
W960_VOLL = ("TS_PLAN npoints=4341 npacked=9424311 numsteps=88200 numfreq=44100 numoutsteps=88146 ngrid=4340 ny=124 nz=35 "
             "ntwr=0 usr=0 scmod=2,0,0 cohgleich=0 wr=100000000 turbmodel=1 desclen=96")
STD120 = ("TS_PLAN npoints=1225 npacked=750925 numsteps=2898 numfreq=1449 numoutsteps=2891 ngrid=1225 ny=35 nz=35 "
          "ntwr=0 usr=0 scmod=2,0,0 cohgleich=0 wr=100000000 turbmodel=1 desclen=96")
W340_120 = ("TS_PLAN npoints=1541 npacked=1188111 numsteps=3024 numfreq=1512 numoutsteps=3019 ngrid=1540 ny=44 nz=35 "
            "ntwr=0 usr=0 scmod=2,0,0 cohgleich=0 wr=100000000 turbmodel=1 desclen=96")
FARMLOW = ("TS_PLAN npoints=2413 npacked=2912491 numsteps=84 numfreq=42 numoutsteps=84 ngrid=2413 ny=127 nz=19 "
           "ntwr=0 usr=0 scmod=2,2,2 cohgleich=1 wr=100000000 turbmodel=1 desclen=96")
FARMHIGH = ("TS_PLAN npoints=3600 npacked=6481800 numsteps=748 numfreq=374 numoutsteps=748 ngrid=3599 ny=59 nz=61 "
            "ntwr=0 usr=1 scmod=2,2,2 cohgleich=1 wr=100000000 turbmodel=15 desclen=96")


def box(zeile, laufdir=Path("/nicht/da")):
    return gw.Box(Path("/x/box.inp"), laufdir, gw.plan_parsen(zeile), timestep=0.05, uref=11.0)


# --- Groessenmodelle --------------------------------------------------------------------------------------------------
def test_bts_groesse_aufs_byte_wie_bestand():
    # Groessen der Referenz-.bts (Datenbytes wie in examples/REFERENCE_SHA256.txt + 70 + 96 Kopfbytes)
    assert box(STD_VOLL).bts_bytes() == 639450166                 # examples/std_full
    assert box(W960_VOLL).bts_bytes() == 2295322006               # examples/w960_full
    assert box(STD120).bts_bytes() == 21249016                    # examples/std120


def test_ram_modell_deckt_gemessenes_rss_ohne_verschwendung():
    # std voll, GPU-Binary, B = 8 (protokoll_g3.log): 5 184 608 KB RSS
    gemessen = 5184608 * 1024
    schaetzung = box(STD_VOLL).ram(8)
    assert gemessen <= schaetzung <= 1.25 * gemessen
    # W960 voll: V(88200 x 4341 x 3) + S + Phasen = 17,1 GiB exakt; mit Rest knapp 18 GiB -> nur allein, passt aber
    # in die 18,9 GiB, die am 17.10. nach 4 GiB Reserve nutzbar waren
    assert 17.3 * GIB < box(W960_VOLL).ram(8) < 18.9 * GIB


@pytest.mark.parametrize("zeile,b,gemeldet_gib", [(STD120, 8, 1.63), (W340_120, 8, 2.55), (FARMLOW, 2, 1.59)])
def test_vram_modell_gegen_tsg_host_meldung(zeile, b, gemeldet_gib):
    """tsg_host meldet 'VRAM x GiB' (eigene Puffer); das Modell ohne HIP-Kontext muss knapp darueber liegen."""
    bx = box(zeile)
    ohne_kontext = (bx.vram(b) - gw.VRAM_KONTEXT) / GIB
    assert gemeldet_gib - 0.01 <= ohne_kontext <= gemeldet_gib * 1.1 + 0.02


def test_komponenten_und_gpu_faehigkeit():
    assert box(STD_VOLL).gpu_aufrufe == 1 and box(STD_VOLL).ncomp == 1
    assert box(FARMHIGH).gpu_aufrufe == 1 and box(FARMHIGH).ncomp == 3          # u,v,w gemeinsam
    assert box(FARMHIGH).gpu_faehig() == ""                                      # ein Nutzerpunkt geht auf der GPU
    general = STD_VOLL.replace("scmod=2,0,0", "scmod=1,0,0")
    assert "GENERAL" in box(general).gpu_faehig()
    zwei_usr = FARMHIGH.replace("usr=1", "usr=2")
    assert "Nutzerpunkte" in box(zwei_usr).gpu_faehig()
    getrennt = FARMLOW.replace("cohgleich=1", "cohgleich=0")
    assert box(getrennt).gpu_aufrufe == 3 and box(getrennt).ncomp == 1
    keine = STD_VOLL.replace("scmod=2,0,0", "scmod=0,0,0")
    assert box(keine).gpu_aufrufe == 0 and box(keine).gpu_faehig() == ""


def test_vram_anteil_verkleinert_b_und_lehnt_ab():
    bx = box(W960_VOLL)
    b8 = bx.vram(8)
    assert gw.vram_anteil(bx, b8) == (8, b8)
    b, anteil = gw.vram_anteil(bx, b8 - 1)
    assert b == 7 and anteil == bx.vram(7)
    assert gw.vram_anteil(bx, bx.vram(1) - 1) == (0, 0)
    assert box(FARMLOW).b_max() == 2                                             # 42 Frequenzen = 2 Gruppen


# --- Planung -----------------------------------------------------------------------------------------------------------
def test_plan_w960_allein_und_std_zu_zweit():
    budget = gw.Budget(ram=int(19.2 * GIB), vram=int(28.8 * GIB), platte=500 * GIB, parallel=2)
    boxen = [box(W960_VOLL), box(STD_VOLL), box(STD_VOLL)]
    pl = gw.plan_erstellen(boxen, budget, 2, {})
    assert pl["machbar"] == 3 and not pl["nicht_ram"] and not pl["nicht_vram"]
    assert pl["gleichzeitig"] == 2 and pl["ram_k"] == 2                          # beide std zusammen, W960 nicht dazu
    assert gw.max_gleichzeitig(box(W960_VOLL), budget)[0] == 1                   # 19,1 von 19,2 GiB: nur allein
    assert gw.max_gleichzeitig(box(STD_VOLL), budget)[0] == 3


def test_plan_meldet_zu_grosse_box_und_platte():
    budget = gw.Budget(ram=10 * GIB, vram=28 * GIB, platte=int(1.0 * GIB), parallel=2)
    boxen = [box(W960_VOLL), box(STD_VOLL), box(STD_VOLL)]
    pl = gw.plan_erstellen(boxen, budget, 2, {})
    assert [b.plan["npoints"] for b in pl["nicht_ram"]] == [4341]
    assert pl["platte_passen"] == 1                                              # 2 x 0,6 GiB > 1 GiB
    eng = gw.Budget(ram=100 * GIB, vram=int(2.5 * GIB), platte=500 * GIB, parallel=2)
    assert [b.plan["npoints"] for b in gw.plan_erstellen([box(W960_VOLL)], eng, 2, {})["nicht_vram"]] == [4341]


def test_schaetzung_nutzt_historie_vor_kalibrierung():
    bx = box(STD120)
    g, art = gw.gpu_schaetzung(bx, {})
    assert art == "grob" and 0.3 < g < 1.2                                       # Kalibrierpunkt std120: 0,59 s
    g, art = gw.gpu_schaetzung(bx, {bx.signatur(): {"gpu_s": 0.7, "wand_s": 1.0, "n": 1}})
    assert (g, art) == (0.7, "gemessen")


def test_fertig_vorhanden_braucht_groesse_und_passende_eingabe(tmp_path):
    inp = tmp_path / "box.inp"
    inp.write_text(KOPF)
    vor_ort = gw.Box(inp, tmp_path, gw.plan_parsen(STD120))
    bts = tmp_path / "box.bts"
    assert not vor_ort.fertig_vorhanden()
    bts.write_bytes(b"\0" * (vor_ort.bts_bytes() - 1))
    assert not vor_ort.fertig_vorhanden()
    with bts.open("ab") as f:
        f.write(b"\0")
    assert vor_ort.fertig_vorhanden()                       # Bestandsbox neben ihrer Eingabe, juenger als sie
    os.utime(inp, (bts.stat().st_mtime + 10, bts.stat().st_mtime + 10))
    assert not vor_ort.fertig_vorhanden()                   # Eingabe danach geaendert
    gespiegelt = tmp_path / "ziel"
    gespiegelt.mkdir()
    shutil.copy(bts, gespiegelt / "box.bts")
    bx = gw.Box(inp, gespiegelt, gw.plan_parsen(STD120))
    assert not bx.fertig_vorhanden()                        # ohne Quittung
    quittung = gespiegelt / "box.gpu.ok"
    quittung.write_text(json.dumps({"eingabe": gw.eingabe_hash(bx), "bts": bx.bts_bytes()}))
    assert bx.fertig_vorhanden()
    inp.write_text(KOPF + "# anderer Seed\n")
    assert not bx.fertig_vorhanden()                        # Quittung passt nicht mehr zur Eingabe


# --- Eingaben ----------------------------------------------------------------------------------------------------------
KOPF = "---------TurbSim v2.00.* Input File------------------------\nTitel\n"


def test_eingabesuche_und_bezogene_dateien(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "_templates").mkdir()
    (tmp_path / "a" / "box1.inp").write_text(KOPF + '"USR.txt"      UserFile    - x\n"unused"  ProfileFile - y\n')
    (tmp_path / "a" / "USR.txt").write_text("1\n")
    (tmp_path / "a" / "_templates" / "vorlage.inp").write_text(KOPF)
    (tmp_path / "a" / "inflow.inp").write_text("------- InflowWind INPUT FILE -------\n")
    (tmp_path / "b.inp").write_text(KOPF + '"../weg.txt"  UserFile - x\n')
    (tmp_path / "weg.txt").write_text("1\n")
    gefunden = gw.finde_eingaben([str(tmp_path / "a")])
    assert [p.name for p in gefunden] == ["box1.inp"]
    assert gw.bezogene_dateien(tmp_path / "a" / "box1.inp") == [("USR.txt", tmp_path / "a" / "USR.txt")]
    sub = tmp_path / "s"
    sub.mkdir()
    shutil.copy(tmp_path / "b.inp", sub / "b.inp")
    with pytest.raises(ValueError):
        gw.bezogene_dateien(sub / "b.inp")
    assert gw.eingabewert('0.0500        TimeStep        - Time step\n', "TimeStep") == "0.0500"


def test_log_auswertung():
    text = ("TurbSim-GPU: N = 4341, 44100 Frequenzen, 1 Komponente(n), 1 S-Klassen, B = 8 -> 251.250 s\n"
            "TurbSim-GPU: 900 Stapel (6.1 Gruppen im Mittel), GPU-Zeit 200.0 s, VRAM 19.60 GiB, Waermebremse Tj 92/84 C "
            "Kante 65/60 C: 12 Pausen, 51.0 s gewartet, Tj hoechstens 79 C, Kante hoechstens 64 C\n")
    p = Path(os.environ.get("TMPDIR", "/tmp")) / f"gw_log_{os.getpid()}.log"
    try:
        p.write_text(text)
        e = gw.log_auswerten(p)
    finally:
        p.unlink(missing_ok=True)
    assert e["gpu_schleife_s"] == pytest.approx(251.25) and e["bremspausen"] == 12
    assert e["gebremst_s"] == pytest.approx(51.0) and e["gpu_rechen_s"] == pytest.approx(200.0)


# --- Sensoren (nachgebautes sysfs) -------------------------------------------------------------------------------------
def test_gpu_suche_und_sensoren(tmp_path):
    drm = tmp_path / "drm"
    for karte, vram, dev in (("card1", 34208743424, "0000:03:00.0"), ("card2", 536870912, "0000:7a:00.0")):
        pci = tmp_path / "pci" / dev
        hw = pci / "hwmon" / "hwmon3"
        hw.mkdir(parents=True)
        (pci / "vendor").write_text("0x1002\n")
        (pci / "mem_info_vram_total").write_text(f"{vram}\n")
        (pci / "mem_info_vram_used").write_text("59924480\n")
        for k, lab, wert in ((1, "edge", 31000), (2, "junction", 77000), (3, "mem", 40000)):
            (hw / f"temp{k}_label").write_text(lab + "\n")
            (hw / f"temp{k}_input").write_text(f"{wert}\n")
        (hw / "power1_average").write_text("212000000\n")
        (hw / "power1_cap").write_text("255000000\n")
        (hw / "freq1_input").write_text("3100000000\n")
        (hw / "fan1_input").write_text("2400\n")
        (drm / karte).mkdir(parents=True)
        (drm / karte / "device").symlink_to(pci)
    (drm / "card1-DP-1").mkdir()
    g = gw.finde_gpu(drm)
    assert g is not None and g.pci == "0000:03:00.0" and g.vram_ges == 34208743424
    s = g.sensoren()
    assert (s["tj"], s["kante"], s["speicher"], s["leistung"], s["takt"], s["luefter"]) == (77.0, 31.0, 40.0, 212.0, 3100.0, 2400.0)
    assert g.leistungsgrenze_w() == 255.0 and g.tj_pfad.name == "temp2_input"


# --- mit dem echten Binary, nur TS_PLAN (keine Rechnung) ----------------------------------------------------------------
@pytest.fixture
def binary():
    if gw.binary_pruefen(BIN):
        pytest.skip("GPU-Binary mit TS_PLAN fehlt")
    return BIN


def test_vermessen_mit_ts_plan(binary, tmp_path):
    inp = BEISPIELE / "std120" / "WS_11_TI_12_Seed_1.inp"
    if not inp.exists():
        pytest.skip("std120-Eingabe fehlt")
    shutil.copy(inp, tmp_path / inp.name)
    arbeit = tmp_path / "arbeit"
    arbeit.mkdir()
    plan = gw.vermessen(binary, tmp_path / inp.name, arbeit)
    assert plan == gw.plan_parsen(STD120)
    assert list(arbeit.iterdir()) == []                                          # Wegwerfordner geraeumt
    assert sorted(p.name for p in tmp_path.iterdir()) == ["WS_11_TI_12_Seed_1.inp", "arbeit"]   # nichts neben der Eingabe


def test_nur_planen_schreibt_nichts(binary, tmp_path, capsys):
    inp = BEISPIELE / "std120" / "WS_11_TI_12_Seed_1.inp"
    if not inp.exists():
        pytest.skip("std120-Eingabe fehlt")
    ein = tmp_path / "ein"
    ein.mkdir()
    shutil.copy(inp, ein / inp.name)
    ziel = tmp_path / "ziel"
    rc = gw.main([str(ein), "--ziel", str(ziel), "--nur-planen"])
    out = capsys.readouterr().out
    assert rc == 0 and "zu rechnen: 1 Boxen" in out and "gleichzeitig:" in out
    assert list(ziel.iterdir()) == []


# --- Ablauf mit einem Ersatz-TurbSim (Shell-Skript, keine GPU) ----------------------------------------------------------
ERSATZ = r'''#!/bin/bash
# Marken fuer binary_pruefen: tsg_pflicht TS_PLAN TS_GPU_TJ_MAX
inp="$1"; stamm="${inp%.inp}"
if [ -n "$TS_PLAN" ] && [ "$TS_PLAN" != 0 ]; then
  echo "TS_PLAN npoints=10 npacked=55 numsteps=20 numfreq=10 numoutsteps=20 ngrid=9 ny=3 nz=3 ntwr=0 usr=0 scmod=2,0,0 cohgleich=0 wr=100000000 turbmodel=1 desclen=6"
  echo " TurbSim terminated normally."; exit 0
fi
if grep -q KAPUTT "$inp"; then echo "ERROR: kaputt"; exit 1; fi
if grep -q HEISS "$inp"; then                     # Karte ueberhitzt; ein Hintergrundprozess kuehlt sie nach 1,5 s wieder
  echo 99000 > "$FAKE_TJ"; ( sleep 1.5; echo 40000 > "$FAKE_TJ" ) & sleep 30; exit 0
fi
if [ -n "$HSA_OVERRIDE_GFX_VERSION$HIP_VISIBLE_DEVICES$ROCR_VISIBLE_DEVICES" ]; then
  echo "TurbSim-GPU: kein gfx1201 -> abgelehnt (TS_GPU=nur: CPU-Pfad gesperrt)" >&2; echo "ERROR: GPU"; exit 1
fi
echo "$PWD $(date +%s.%N) start" >> "$FAKE_SPUR"
sleep "${FAKE_DAUER:-0.6}"
head -c $((70 + 6 + 6 * 9 * 20)) /dev/zero > "$stamm.bts"
echo "    Frequenzschleife auf der GPU gerechnet"
echo "TurbSim-GPU: N = 10, 10 Frequenzen, 1 Komponente(n), 1 S-Klassen, B = 1 -> 0.100 s" >&2
echo "TurbSim-GPU: 1 Stapel (1.0 Gruppen im Mittel), GPU-Zeit 0.090 s, VRAM 0.01 GiB, Waermebremse 78/70 C: 0 Pausen, 0.0 s gewartet, Tj hoechstens 40 C" >&2
echo "  Processing complete.  0.1 CPU seconds used."
echo "$PWD $(date +%s.%N) ende" >> "$FAKE_SPUR"
'''


@pytest.fixture
def ersatz(tmp_path, monkeypatch):
    """Ersatz-Binary, nachgebaute GPU (kuehl), Historie und Sperre im tmp, Spur der Starts/Enden."""
    binary = tmp_path / "turbsim_ersatz"
    binary.write_text(ERSATZ)
    binary.chmod(0o755)
    spur = tmp_path / "spur.txt"
    monkeypatch.setenv("FAKE_SPUR", str(spur))
    monkeypatch.setattr(gw, "HISTORIE", tmp_path / "cache" / "laufzeiten.json")
    drm = tmp_path / "drm"
    pci = tmp_path / "pci" / "0000:03:00.0"
    hw = pci / "hwmon" / "hwmon3"
    hw.mkdir(parents=True)
    for name, wert in (("vendor", "0x1002"), ("mem_info_vram_total", str(32 * GIB)), ("mem_info_vram_used", "0")):
        (pci / name).write_text(wert + "\n")
    (hw / "temp2_label").write_text("junction\n")
    (hw / "temp2_input").write_text("40000\n")
    (hw / "temp1_label").write_text("edge\n")
    (hw / "temp1_input").write_text("35000\n")
    monkeypatch.setenv("FAKE_TJ", str(hw / "temp2_input"))
    (hw / "power1_cap").write_text("255000000\n")
    (drm / "card1").mkdir(parents=True)
    (drm / "card1" / "device").symlink_to(pci)
    gpu = gw.finde_gpu(drm)
    monkeypatch.setattr(gw, "finde_gpu", lambda: gpu)
    monkeypatch.setattr(gw, "fremde_gpu_nutzer", lambda pci, ausser=frozenset(): [])
    return dict(binary=binary, spur=spur, tmp=tmp_path, hw=hw)


def _eingabe(d: Path, name: str, extra: str = "") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(KOPF + "0.0500        TimeStep        - x\n11.0   URef  - x\n" + extra)
    return p


def _spur(spur: Path) -> list[tuple[str, float, str]]:
    if not spur.exists():
        return []
    return [(a, float(b), c) for a, b, c in (z.split() for z in spur.read_text().splitlines())]


def test_lauf_quittung_und_wiederaufnahme(ersatz, monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "12.0.0")                    # wie in der .bashrc (Dauertest 18.10.)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    ein, ziel = ersatz["tmp"] / "ein", ersatz["tmp"] / "ziel"
    _eingabe(ein / "a", "box1.inp")
    _eingabe(ein / "b", "box1.inp")                                              # gleicher Name, anderer Ordner
    argv = [str(ein), "--ziel", str(ziel), "--binary", str(ersatz["binary"]), "--trotz-warnung"]
    assert gw.main(argv) == 0
    for sub in ("a", "b"):
        assert (ziel / sub / "box1.bts").stat().st_size == 70 + 6 + 6 * 9 * 20
        assert json.loads((ziel / sub / "box1.gpu.ok").read_text())["bts"] == 70 + 6 + 6 * 9 * 20
    n = len(_spur(ersatz["spur"]))
    assert gw.main(argv) == 0 and len(_spur(ersatz["spur"])) == n             # fertig: nichts neu gerechnet
    (ein / "a" / "box1.inp").write_text((ein / "a" / "box1.inp").read_text() + "# geaendert\n")
    assert gw.main(argv) == 0 and len(_spur(ersatz["spur"])) == n + 2         # nur die geaenderte Box neu
    assert gw.finde_eingaben([str(ersatz["tmp"])], ziel) == sorted(gw.finde_eingaben([str(ein)]))  # Kopien im Ziel nicht


def test_dauertest_startet_box_nie_doppelt_und_schont_ziel(ersatz):
    ein, ziel = ersatz["tmp"] / "ein", ersatz["tmp"] / "ziel"
    _eingabe(ein, "box1.inp")
    ziel.mkdir()
    alt = ziel / "box1.bts"
    alt.write_bytes(b"vorhandenes Feld")
    rc = gw.main([str(ein), "--ziel", str(ziel), "--binary", str(ersatz["binary"]), "--trotz-warnung",
                  "--dauertest", "0.05", "--parallel", "2"])
    assert rc == 0
    assert alt.read_bytes() == b"vorhandenes Feld"                               # nichts Vorhandenes geloescht
    spur = _spur(ersatz["spur"])
    starts = [t for _, t, a in spur if a == "start"]
    enden = [t for _, t, a in spur if a == "ende"]
    assert len(starts) >= 2 and len(starts) == len(enden)
    for s_neu, e_alt in zip(starts[1:], enden):                                  # jede Wiederholung erst nach dem Ende der vorigen
        assert s_neu >= e_alt
    assert all("dauertest_" in d for d, _, _ in spur)
    berichte = list(ziel.glob("dauertest_*/bericht_*.txt"))
    assert len(berichte) == 1 and f"fertig: {len(starts)}" in berichte[0].read_text()


def test_fehlerserie_haelt_an_und_tsplan_aus_der_shell_stoert_nicht(ersatz, monkeypatch):
    monkeypatch.setenv("TS_PLAN", "1")                                           # uebriggeblieben in der Shell
    ein, ziel = ersatz["tmp"] / "ein", ersatz["tmp"] / "ziel"
    _eingabe(ein, "a1.inp")                                                      # groesste zuerst: gleiche Groesse -> Reihenfolge
    for name in ("k1.inp", "k2.inp", "k3.inp"):
        _eingabe(ein, name, "KAPUTT\n")
    rc = gw.main([str(ein), "--ziel", str(ziel), "--binary", str(ersatz["binary"]), "--trotz-warnung", "--parallel", "1"])
    assert rc == 1
    bericht = next(ziel.glob("bericht_*.txt")).read_text()
    assert "nicht gestartet: zwei Boxen nacheinander gescheitert" in bericht
    assert (ziel / "a1.bts").exists() or "fertig" in bericht                     # die heile Box rechnet trotz TS_PLAN


def test_waehle_b_beachtet_ram():
    bx = box(W960_VOLL)
    b, anteil = gw.waehle_b(bx, 28 * GIB, bx.ram(8) - 1)
    assert 1 <= b < 8 and anteil == bx.vram(b) and bx.ram(b) <= bx.ram(8) - 1
    assert gw.waehle_b(bx, 28 * GIB, bx.ram(1) - 1) == (0, 0)


def test_vorbereiten_loescht_keine_gleichnamige_nutzerdatei(tmp_path):
    ein = tmp_path / "ein"
    _eingabe(ein, "box.inp", '"box.dat"  UserFile  - x\n')
    (ein / "box.dat").write_text("Zeitreihe\n")
    bx = gw.Box(ein / "box.inp", ein, gw.plan_parsen(STD120))                    # vor Ort
    gw.box_vorbereiten(bx)
    assert (ein / "box.dat").read_text() == "Zeitreihe\n"


def test_waechter_beendet_ueberhitzte_box_und_wiederholt_einmal(ersatz):
    ein, ziel = ersatz["tmp"] / "ein", ersatz["tmp"] / "ziel"
    _eingabe(ein, "heiss.inp", "HEISS\n")
    t0 = time.time()
    rc = gw.main([str(ein), "--ziel", str(ziel), "--binary", str(ersatz["binary"]), "--trotz-warnung"])
    assert rc == 1 and time.time() - t0 < 25                                     # beendet, nicht 2 x 30 s abgewartet
    bericht = next(ziel.glob("bericht_*.txt")).read_text()
    assert "gescheitert" in bericht and "vom Waechter beendet: Tj 99 C" in bericht
    assert not (ziel / "heiss.bts").exists()

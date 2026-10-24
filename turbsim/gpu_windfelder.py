#!/usr/bin/env python3
"""TurbSim-Windfelder NUR auf der GPU erzeugen, mit Ressourcenplanung, damit nichts abstuerzt (17.10.2026).

Ablauf
  1. System erfassen: RAM (MemAvailable), Swap, VRAM der R9700, Leistungsgrenze, Temperaturen, freie Platte, Fremdlast.
  2. Jede Box mit TS_PLAN vermessen (TurbSims eigene Gitterlogik, ~10 ms, kein Speicher): Punkte, Zeitschritte,
     Frequenzen, Ausgabegitter. Daraus RAM (V, S, Phasen, Kohaerenz-Hilfsfelder), VRAM je Frequenzgruppe und
     .bts-Groesse (aufs Byte).
  3. Plan: wie viele Boxen gleichzeitig (RAM, VRAM, GPU-Ueberlappung), wie viele insgesamt auf die Platte passen, Dauer.
  4. Lauf (ohne --nur-planen):
       * TS_GPU=nur — die Frequenzschleife rechnet nur die GPU; kann sie nicht, bricht die Box ab (kein CPU-Rueckfall).
         Die CPU macht nur Einlesen, Zufallsphasen, FFT und Ausgabe, mit --cpu-faeden Faeden (Vorgabe 2, kein AVX-512).
       * Waermebremse im Binary (TS_GPU_TJ_MAX/WIEDER, Stapel von ~TS_GPU_STAPEL_S Sekunden), Profil --profil;
         ohne lesbaren Sensor Ersatzbetrieb mit 50 % Pause statt ungebremst.
       * Speicherdeckel je Box (systemd-run --scope, MemoryMax, kein Swap): reicht die Schaetzung nicht, stirbt nur
         diese Box, nicht der Rechner; RAM-Notbremse, wenn der freie RAM trotzdem unter --ram-not-gb faellt.
       * Waechter: liegt die Sperrschicht zweimal in Folge ueber der Notfallgrenze, werden die Boxen beendet (die GPU
         stoppt sofort), die Karte kuehlt ab, die Box kommt einmal erneut in die Warteschlange.
       * Halt nach einem GPU-Fehler oder zwei gescheiterten Boxen in Folge; keine neuen Boxen, solange fremde Last auf
         der GPU liegt; nur ein Laeufer zur Zeit (Sperrdatei); stirbt der Laeufer, sterben seine Boxen mit.
       * Sensorprotokoll (CSV, fsync je Zeile) und Bericht im Zielordner; fertige Boxen (Quittung <stamm>.gpu.ok mit
         dem Hash der Eingabe) werden uebersprungen.

Die Rechnung ist bitgleich zu /usr/local/bin/turbsim (v3.5.3): Planung, Bremse, Stapelgroesse und Parallelitaet
aendern kein Bit (jede Frequenz ist eine eigene Kette, turbsim/gpu/tsg_host.hip).

Beispiele
  python3 turbsim/gpu_windfelder.py examples --ziel out --nur-planen
  python3 turbsim/gpu_windfelder.py meine_boxen/*.inp --ziel /pfad/ausgabe
  python3 turbsim/gpu_windfelder.py examples/w960_full --ziel out --dauertest 30
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN_VORGABE = ROOT / "build" / "turbsim-gpu" / "modules" / "turbsim" / "turbsim"
HISTORIE = Path.home() / ".cache" / "turbsim-gpu" / "laufzeiten.json"
GIB = 1 << 30
TSG_L = 32                      # Frequenzen je Gruppe (tsg_chol.hip)
TSG_W = 8                       # Blockbreite der GPU-Cholesky
TSG_MAXM = 256
B_HOECHST = 8                   # tsg_host.hip: hoechstens 8 Gruppen je Stapel

# Waermebremse und Waechter. Beobachtet (messprotokoll.md): Haenger bei Tj 93-94 C nach ~2 min Volllast (255 W, 14.10.).
# Dauertest 18.10. (255 W): der Hotspot (Tj) folgt der Leistung binnen ~1 s, ~28 C ueber der Kante bei 250 W, und faellt in
# jeder Pause sofort -> eine Tj-Grenze vor jedem Stapel laesst die Karte zwischen 66 und 96 C pendeln. Geregelt wird darum
# die traege Kantentemperatur (kante_max/kante_wieder). Gemessen: Tj liegt unter Last 24-31 C ueber der Kante, bei 255 W
# wie bei 210 W (18.10., Farm-High bei 210 W: Kante 37 C, Tj 66-68 C) -> Kantengrenze = Notfallgrenze - 30 C - Rand.
# Der Waechter beendet im Notfall die Boxen (SIGKILL stoppt die GPU sofort; SIGSTOP liess die schon eingereihte Arbeit
# noch 4-6 s mit voller Leistung laufen). leistung_max: hoechste erlaubte Leistungsgrenze der Karte (LACT) fuer das Profil.
PROFILE = {
    "sicher": dict(kante_max=58.0, kante_wieder=53.0, tj_max=92.0, tj_wieder=84.0, stapel_s=6.0, notfall=93.0,
                   leistung_max=215.0),
    "normal": dict(kante_max=62.0, kante_wieder=57.0, tj_max=94.0, tj_wieder=86.0, stapel_s=6.0, notfall=95.0,
                   leistung_max=255.0),
    "voll":   dict(kante_max=66.0, kante_wieder=61.0, tj_max=96.0, tj_wieder=88.0, stapel_s=6.0, notfall=97.0,
                   leistung_max=0.0),
}
# stapel_s 6 s (18.10.): mit dem beendenden Waechter muss die eingereihte Arbeit nicht mehr kurz sein; 2-s-Stapel kosteten
# beim W960-Dauertest ~40 % GPU-Zeit (340 statt ~240 s je Box). Die Kante steigt unter Last ~0,15 C/s -> <= 1 C Ueberschwingen.
NOTFALL_MESSUNGEN = 2           # so viele Waechter-Messungen in Folge (je 0,5 s) ueber der Notfallgrenze -> beenden
NOTFAELLE_MAX = 3               # so viele Notabschaltungen je Lauf, dann Halt (Leistungsgrenze/Luefter pruefen)
LEISTUNG_GETESTET_W = 255

# Gemessene GPU-Schleifenzeiten (17.10.2026, R9700, 255 W, Binary build-ts353-gpu) fuer die Erstschaetzung:
# (Punkte, Frequenzen, TimeStep, GPU-Sekunden). Spaetere Laeufe ueberschreiben das ueber die Historie.
KALIBRIERUNG = [
    (1225, 1449, 0.05, 0.59),     # std120
    (1225, 43520, 0.05, 16.4),    # std voll (14.10., aelterer Kern)
    (1541, 1512, 0.05, 0.96),     # W340-120
    (4341, 2079, 0.05, 11.2),     # W960-120
    (3600, 374, 0.42, 6.2),       # Farm-High (u,v,w gemeinsam)
    (2413, 42, 3.78, 0.65),       # Farm-Low
]
CPU_S_JE_PUNKTSCHRITT = 8e-8    # Einlesen, Phasen, FFT, Ausgabe mit 2 Faeden (std voll: ~5,7 s bei 4 Faeden)
RAM_SOCKEL = 300 << 20          # Programm, HIP-Laufzeit, Puffer ausserhalb des Modells (std voll gemessen: ~140 MB)
RAM_ZUSCHLAG = 1.25             # nur auf den unsicheren Teil (Sockel, Hilfsfelder, Puffer); V, S, Phasen sind exakt
VRAM_KONTEXT = 512 << 20        # HIP-Kontext, Code-Objekte, Scratch je Prozess (Schaetzung)
# Variablen, die die Karte verstecken oder umdeuten (HSA_OVERRIDE_GFX_VERSION=12.0.0 aus der .bashrc gibt die R9700 als
# gfx1200 aus -> "kein gfx1201", Dauertest 18.10.); fuer TurbSim immer entfernt, das Binary waehlt gfx1201 selbst.
GPU_SICHT_VARIABLEN = ("HSA_OVERRIDE_GFX_VERSION", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                       "GPU_DEVICE_ORDINAL")


# --------------------------------------------------------------------------------------------------------------------
# System
# --------------------------------------------------------------------------------------------------------------------
def _lies(pfad: Path | str, vorgabe: str = "") -> str:
    try:
        return Path(pfad).read_text().strip()
    except OSError:
        return vorgabe


def _lies_int(pfad: Path | str, vorgabe: int = -1) -> int:
    try:
        return int(_lies(pfad, str(vorgabe)))
    except ValueError:
        return vorgabe


def meminfo() -> dict[str, int]:
    """Bytes aus /proc/meminfo (MemTotal, MemAvailable, SwapTotal, SwapFree, ...)."""
    werte = {}
    for zeile in _lies("/proc/meminfo").splitlines():
        teile = zeile.split()
        if len(teile) >= 2 and teile[1].isdigit():
            werte[teile[0].rstrip(":")] = int(teile[1]) * (1024 if len(teile) > 2 and teile[2] == "kB" else 1)
    return werte


@dataclass
class Gpu:
    karte: Path                 # /sys/class/drm/cardN/device
    pci: str
    vram_ges: int
    hwmon: Path | None
    tj_pfad: Path | None

    def vram_belegt(self) -> int:
        return max(0, _lies_int(self.karte / "mem_info_vram_used", 0))

    def sensoren(self) -> dict[str, float]:
        """Tj/Kante/Speicher in C, Leistung in W, Takt in MHz, Luefter in rpm; fehlende Werte als -1."""
        w = dict(tj=-1.0, kante=-1.0, speicher=-1.0, leistung=-1.0, takt=-1.0, luefter=-1.0)
        if self.hwmon is None:
            return w
        for k in range(1, 9):
            lab = _lies(self.hwmon / f"temp{k}_label")
            if lab in ("junction", "edge", "mem"):
                v = _lies_int(self.hwmon / f"temp{k}_input")
                w[{"junction": "tj", "edge": "kante", "mem": "speicher"}[lab]] = v / 1000.0 if v >= 0 else -1.0
        for name in ("power1_average", "power1_input"):
            v = _lies_int(self.hwmon / name)
            if v >= 0:
                w["leistung"] = v / 1e6
                break
        v = _lies_int(self.hwmon / "freq1_input")
        w["takt"] = v / 1e6 if v >= 0 else -1.0
        v = _lies_int(self.hwmon / "fan1_input")
        w["luefter"] = float(v)
        return w

    def leistungsgrenze_w(self) -> float:
        v = _lies_int(self.hwmon / "power1_cap") if self.hwmon else -1
        return v / 1e6 if v > 0 else -1.0

    def wecken(self) -> int:
        """Render-Knoten der Karte oeffnen: amdgpu weckt sie aus dem Runtime-Suspend (sonst liefert hwmon nichts).
        Rueckgabe: Dateideskriptor (mit os.close schliessen) oder -1."""
        for r in sorted(Path("/sys/class/drm").glob("renderD*")):
            if os.path.realpath(r / "device") == os.path.realpath(self.karte):
                try:
                    fd = os.open(f"/dev/dri/{r.name}", os.O_RDWR)
                except OSError:
                    return -1
                for _ in range(20):                     # bis hwmon antwortet (hoechstens ~2 s)
                    if self.sensoren()["tj"] >= 0:
                        break
                    time.sleep(0.1)
                return fd
        return -1


def finde_gpu(drm: Path = Path("/sys/class/drm")) -> Gpu | None:
    """Die AMD-Karte mit dem groessten VRAM (die R9700; die iGPU hat 0,5 GB)."""
    beste = None
    for karte in sorted(drm.glob("card[0-9]*")):
        if "-" in karte.name:
            continue
        dev = karte / "device"
        if _lies(dev / "vendor") != "0x1002":
            continue
        vram = _lies_int(dev / "mem_info_vram_total", 0)
        if vram < 4 * GIB or (beste and vram <= beste.vram_ges):
            continue
        hw = next(iter(sorted((dev / "hwmon").glob("hwmon*"))), None) if (dev / "hwmon").is_dir() else None
        tj = None
        if hw is not None:
            for k in range(1, 9):
                if _lies(hw / f"temp{k}_label") == "junction":
                    tj = hw / f"temp{k}_input"
        beste = Gpu(dev, Path(os.path.realpath(dev)).name, vram, hw, tj)
    return beste


def cpu_tctl() -> float:
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if _lies(hw / "name") == "k10temp":
            v = _lies_int(hw / "temp1_input")
            return v / 1000.0 if v >= 0 else -1.0
    return -1.0


def cpu_name() -> str:
    for zeile in _lies("/proc/cpuinfo").splitlines():
        if zeile.startswith("model name"):
            return zeile.split(":", 1)[1].strip()
    return "?"


def fremde_gpu_nutzer(pci: str, ausser: set[int] | frozenset = frozenset()) -> list[str]:
    """Prozesse (des Nutzers), die die Karte offen haben und VRAM belegen (fdinfo drm-pdev/drm-memory-vram)."""
    gefunden = {}
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid() or int(proc.name) in ausser:
            continue
        try:
            for fd in (proc / "fdinfo").iterdir():
                txt = fd.read_text()
                if f"drm-pdev:\t{pci}" not in txt:
                    continue
                m = re.search(r"drm-memory-vram:\s+(\d+)\s*KiB", txt)
                kib = int(m.group(1)) if m else 0
                if kib >= 64 * 1024:
                    gefunden[proc.name] = max(gefunden.get(proc.name, 0), kib)
        except (OSError, PermissionError):
            continue
    return [f"{_lies(Path('/proc') / pid / 'comm', '?')} (PID {pid}, {kib / 1048576:.1f} GiB)" for pid, kib in gefunden.items()]


def laufende_rechner() -> list[str]:
    namen = ("turbsim", "openfast", "FAST.Farm", "interFoam", "simpleFoam")
    treffer = []
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit():
            comm = _lies(proc / "comm")
            if any(comm.startswith(n) for n in namen):
                treffer.append(f"{comm} (PID {proc.name})")
    return treffer


# --------------------------------------------------------------------------------------------------------------------
# Boxen
# --------------------------------------------------------------------------------------------------------------------
def ist_turbsim_eingabe(p: Path) -> bool:
    try:
        kopf = p.open("rb").read(300).decode("latin-1")
    except OSError:
        return False
    return "TurbSim" in kopf.splitlines()[0] if kopf else False


def finde_eingaben(argumente: list[str], ziel: Path | None = None) -> list[Path]:
    """TurbSim-Eingaben aus Dateien und Ordnern (rekursiv). Liegt der Zielordner in einem Eingabeordner, werden die
    Kopien darin nicht noch einmal als Boxen gefunden (ausser Ziel = Eingabeordner: Rechnen vor Ort)."""
    ziel = ziel.resolve() if ziel else None
    gefunden: list[Path] = []
    for a in argumente:
        p = Path(a)
        if p.is_dir():
            kandidaten = sorted(q for q in p.rglob("*.inp") if "_templates" not in q.parts)
            pr = p.resolve()
            if ziel and ziel != pr and pr in ziel.parents:
                kandidaten = [q for q in kandidaten if ziel not in q.resolve().parents]
        else:
            kandidaten = [p]
        for q in kandidaten:
            q = q.resolve()
            if q.suffix.lower() == ".inp" and q.is_file() and ist_turbsim_eingabe(q) and q not in gefunden:
                gefunden.append(q)
    return gefunden


def eingabewert(text: str, name: str) -> str | None:
    """Wert vor dem Parameternamen (OpenFAST-Format: <wert>  <Name>  - Beschreibung)."""
    m = re.search(rf"^\s*(\"[^\"]*\"|\S+)\s+{re.escape(name)}\b", text, re.M)
    return m.group(1).strip('"') if m else None


def bezogene_dateien(inp: Path) -> list[tuple[str, Path]]:
    """Dateien, die TurbSim zur Box liest (UserFile, ProfileFile): (Name wie in der Eingabe, Pfad).

    Absolute Namen liest TurbSim von dort und brauchen keine Kopie (Name leer). Relative Namen, die aus dem
    Eingabeordner herausfuehren ("../x"), lassen sich im gespiegelten Ordner nicht nachbilden -> ValueError.
    """
    text = inp.read_text(errors="replace")
    dateien = []
    for name in ("UserFile", "ProfileFile"):
        w = eingabewert(text, name)
        if not w or w.lower() in ("unused", "none", "default"):
            continue
        if os.path.isabs(w):
            continue
        rel = os.path.normpath(w)
        q = inp.parent / rel
        if not q.is_file():
            continue
        if rel.startswith(".."):
            raise ValueError(f"{name} '{w}' liegt ausserhalb des Eingabeordners - bitte absolut angeben")
        dateien.append((rel, q))
    return dateien


def eingabe_hash(box: "Box") -> str:
    """sha256 ueber die Eingabe und die Nutzerdateien (Quittung einer fertigen Box)."""
    h = hashlib.sha256(box.inp.read_bytes())
    for rel, q in bezogene_dateien(box.inp):
        h.update(rel.encode())
        h.update(q.read_bytes())
    return h.hexdigest()


def plan_parsen(zeile: str) -> dict:
    werte = {}
    for tok in zeile.split()[1:]:
        k, _, v = tok.partition("=")
        if k == "scmod":
            werte[k] = [int(x) for x in v.split(",")]
        elif k == "wr":
            werte[k] = v
        else:
            werte[k] = int(v)
    for k in ("npoints", "npacked", "numsteps", "numfreq", "numoutsteps", "ngrid", "ny", "nz", "ntwr", "usr",
              "scmod", "cohgleich", "wr", "desclen"):
        if k not in werte:
            raise ValueError(f"TS_PLAN-Zeile ohne {k}: {zeile!r}")
    return werte


def vermessen(binary: Path, inp: Path, arbeit: Path) -> dict:
    """TurbSim mit TS_PLAN in einem Wegwerfordner (Verknuepfungen auf Eingabe und Nutzerdateien) laufen lassen."""
    d = Path(tempfile.mkdtemp(prefix="plan_", dir=arbeit))
    try:
        (d / inp.name).symlink_to(inp)
        for rel, q in bezogene_dateien(inp):
            ziel = d / rel
            ziel.parent.mkdir(parents=True, exist_ok=True)
            if not ziel.exists():
                ziel.symlink_to(q.resolve())
        env = dict(os.environ, TS_PLAN="1", TS_GPU="nur", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        r = subprocess.run([str(binary), inp.name], cwd=d, env=env, capture_output=True, text=True, timeout=300)
        zeilen = [z for z in r.stdout.splitlines() if z.startswith("TS_PLAN ")]
        if not zeilen:
            rest = (r.stdout + r.stderr).strip().splitlines()[-4:]
            raise RuntimeError("TurbSim meldet keine Groessen: " + " | ".join(rest))
        return plan_parsen(zeilen[-1])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def skizze_bytes(n: int) -> int:
    ng = (n + TSG_W - 1) // TSG_W
    return (ng * ng + ((ng + 7) // 8) * ng + ng * n) * TSG_L * 2


@dataclass
class Box:
    inp: Path
    laufdir: Path
    plan: dict
    timestep: float = 0.0
    uref: float = 0.0
    anzeige: str = ""                   # Pfad relativ zur gemeinsamen Wurzel der Eingaben
    fehler: str = ""                    # nicht rechenbar (Grund)
    zustand: str = "offen"              # offen, fertig_vorhanden, laeuft, fertig, gescheitert, uebersprungen
    ergebnis: dict = field(default_factory=dict)

    @property
    def stamm(self) -> str:
        return self.inp.stem

    @property
    def name(self) -> str:
        return self.anzeige or self.inp.name

    # --- Groessen aus TurbSims eigener Gitterlogik ---
    @property
    def n(self) -> int:
        return self.plan["npoints"]

    @property
    def iec(self) -> list[int]:
        return [i for i, m in enumerate(self.plan["scmod"]) if m == 2]

    @property
    def ncomp(self) -> int:
        """Komponenten je GPU-Aufruf (u,v,w gemeinsam bei identischen IEC-Parametern)."""
        return 3 if self.plan["cohgleich"] else 1

    @property
    def gpu_aufrufe(self) -> int:
        if not self.iec:
            return 0
        return 1 if self.plan["cohgleich"] else len(self.iec)

    def gpu_faehig(self) -> str:
        """Leer, wenn die Box ohne CPU-Frequenzschleife laeuft; sonst der Grund."""
        sc = self.plan["scmod"]
        if any(m in (1, 3) for m in sc):
            return "Kohaerenzmodell GENERAL/API (rechnet nur die CPU)"
        if self.plan["usr"] > 1 and self.iec:
            return f"{self.plan['usr']} Nutzerpunkte (rechnet nur die CPU)"
        return ""

    def k_schaetzung(self) -> int:
        return min(self.plan["npacked"], 4 * self.n)

    def m_schaetzung(self) -> int:
        return min(TSG_MAXM, self.plan["nz"] + 2 + self.plan["usr"])

    def vram_je_gruppe(self) -> int:
        n, nc, m, k = self.n, self.ncomp, self.m_schaetzung(), self.k_schaetzung()
        return (self.plan["npacked"] * TSG_L * 8 + skizze_bytes(n) + nc * m * (m + 1) * TSG_L * 8
                + 2 * (2 * k * TSG_L * 8 + 2 * nc * n * TSG_L * 8 + nc * m * TSG_L * 8 + nc * n * 2 * TSG_L * 8))

    def vram_fest(self) -> int:
        return self.plan["npacked"] * 4 + self.n * 4 + (64 << 20) + VRAM_KONTEXT

    def b_max(self) -> int:
        return max(1, min(B_HOECHST, (self.plan["numfreq"] + TSG_L - 1) // TSG_L))

    def vram(self, b: int) -> int:
        return self.vram_fest() + b * self.vram_je_gruppe() if self.gpu_aufrufe else VRAM_KONTEXT

    def ram(self, b: int = B_HOECHST) -> int:
        n, ns, nf = self.n, self.plan["numsteps"], self.plan["numfreq"]
        felder = 8 * 3 * n * (ns + 2 * nf)                      # V(NumSteps,N,3), S und Phasen (NumFreq,N,3)
        kohaerenz = 28 * self.plan["npacked"] if self.iec else 0   # Dist, DistU, TRH, Klassen (beruehrte Seiten)
        k, m, nc = self.k_schaetzung(), self.m_schaetzung(), self.ncomp
        pinned = 2 * 8 * b * (2 * k * TSG_L + 2 * nc * n * TSG_L + nc * m * TSG_L + nc * n * 2 * TSG_L) if self.iec else 0
        return int(felder + (kohaerenz + pinned + RAM_SOCKEL) * RAM_ZUSCHLAG)

    def bts_bytes(self) -> int:
        p = self.plan
        if p["wr"][0] != "1":
            return 0
        return 70 + p["desclen"] + 6 * (p["ngrid"] + p["ntwr"]) * p["numoutsteps"]

    def ausgabe_bytes(self) -> int:
        p, wr = self.plan, self.plan["wr"]
        nos, ng = p["numoutsteps"], p["ngrid"]
        summe = self.bts_bytes() + (64 << 10)                  # .sum und Protokoll
        summe += (6 * ng * nos) if wr[1] == "1" else 0          # .wnd
        summe += (130 * nos) if wr[2] == "1" else 0             # .hh
        summe += (60 * nos) if wr[3] == "1" else 0              # .bin
        summe += (130 * nos) if wr[4] == "1" else 0             # .dat
        summe += (3 * 11 * ng * nos) if wr[5] == "1" else 0     # .u/.v/.w (Text)
        summe += (12 * ng * nos) if wr[8] == "1" else 0         # HAWC
        return summe

    def signatur(self) -> str:
        p = self.plan
        return f"n{p['npoints']}_f{p['numfreq']}_y{p['ny']}_z{p['nz']}_dt{self.timestep:g}_u{self.uref:g}_c{p['cohgleich']}"

    def fertig_vorhanden(self) -> bool:
        """.bts exakt so gross wie geplant UND zur aktuellen Eingabe gehoerig: Quittung <stamm>.gpu.ok mit gleichem
        Eingabe-Hash; ohne Quittung nur eine Bestandsbox neben ihrer Eingabe, die juenger ist als die Eingabe."""
        bts = self.laufdir / f"{self.stamm}.bts"
        if not (self.bts_bytes() > 0 and bts.is_file() and bts.stat().st_size == self.bts_bytes()):
            return False
        quittung = self.laufdir / f"{self.stamm}.gpu.ok"
        if quittung.is_file():
            try:
                d = json.loads(quittung.read_text())
                return d.get("eingabe") == eingabe_hash(self) and d.get("bts") == self.bts_bytes()
            except (OSError, ValueError):
                return False
        return self.laufdir.resolve() == self.inp.parent.resolve() and bts.stat().st_mtime >= self.inp.stat().st_mtime


def gpu_schaetzung(box: Box, historie: dict) -> tuple[float, str]:
    """GPU-Sekunden der ganzen Frequenzschleife: aus der Historie, sonst grob aus der Kalibrierung."""
    h = historie.get(box.signatur())
    if h and h.get("gpu_s", 0) > 0:
        return h["gpu_s"], "gemessen"
    if not box.gpu_aufrufe:
        return 0.0, "keine"
    n, nf = box.n, box.plan["numfreq"]

    def abstand(k):
        kn, kf, kdt, _ = k
        return abs(kn - n) / n + 2.0 * abs((kdt - box.timestep) / max(kdt, box.timestep, 1e-9))

    kn, kf, kdt, ks = min(KALIBRIERUNG, key=abstand)
    je_matrix = ks / kf * (n / kn) ** 2.1
    faktor = box.gpu_aufrufe if not box.plan["cohgleich"] else 1
    return je_matrix * nf * faktor, "grob"


def cpu_schaetzung(box: Box, faeden: int) -> float:
    return CPU_S_JE_PUNKTSCHRITT * box.n * box.plan["numsteps"] * 2.0 / max(1, min(faeden, 4))


def lade_historie() -> dict:
    try:
        return json.loads(HISTORIE.read_text())
    except (OSError, ValueError):
        return {}


def speichere_historie(h: dict) -> None:
    try:
        HISTORIE.parent.mkdir(parents=True, exist_ok=True)
        tmp = HISTORIE.with_suffix(".tmp")
        tmp.write_text(json.dumps(h, indent=1, sort_keys=True))
        tmp.replace(HISTORIE)
    except OSError:
        pass


# --------------------------------------------------------------------------------------------------------------------
# Planung
# --------------------------------------------------------------------------------------------------------------------
@dataclass
class Budget:
    ram: int            # fuer TurbSim nutzbar (MemAvailable - Reserve)
    vram: int           # fuer TurbSim nutzbar (VRAM - Reserve - fremd belegt)
    platte: int         # frei - Reserve
    parallel: int       # Obergrenze gleichzeitiger Boxen


def vram_anteil(box: Box, frei_vram: int) -> tuple[int, int]:
    """(Gruppen B, VRAM-Anteil) fuer eine Box bei frei_vram verfuegbarem VRAM; B = 0: passt nicht."""
    bmax = box.b_max()
    for b in range(bmax, 0, -1):
        if box.vram(b) <= frei_vram:
            return b, box.vram(b)
    return 0, 0


def waehle_b(box: Box, vram_frei: int, ram_frei: int) -> tuple[int, int]:
    """Groesste Gruppenzahl b, fuer die VRAM UND RAM reichen (die Puffer wachsen mit b): (b, VRAM-Anteil), (0, 0) wenn
    nicht einmal b = 1 passt. Dieselbe Regel gilt in Plan und Lauf."""
    if not box.gpu_aufrufe:
        return (1, VRAM_KONTEXT) if box.ram(1) <= ram_frei and VRAM_KONTEXT <= vram_frei else (0, 0)
    for b in range(box.b_max(), 0, -1):
        if box.vram(b) <= vram_frei and box.ram(b) <= ram_frei:
            return b, box.vram(b)
    return 0, 0


def max_gleichzeitig(box: Box, budget: Budget) -> tuple[int, int]:
    """Wie viele Boxen dieser Art gleichzeitig in RAM bzw. VRAM passen (VRAM mit mindestens 1 Gruppe je Box)."""
    durch_ram = budget.ram // box.ram(1) if box.ram(1) > 0 else 0
    durch_vram = budget.vram // box.vram(1) if box.vram(1) > 0 else 0
    return int(durch_ram), int(durch_vram)


def plan_erstellen(boxen: list[Box], budget: Budget, faeden: int, historie: dict) -> dict:
    offen = [b for b in boxen if b.zustand == "offen"]
    gross_ram = max((b.ram() for b in offen), default=0)
    nicht_ram = [b for b in offen if b.ram(1) > budget.ram]
    nicht_vram = [b for b in offen if b.gpu_aufrufe and vram_anteil(b, budget.vram)[0] == 0]
    # Platte: in Laufreihenfolge aufsummieren
    summe, passen = 0, 0
    for b in offen:
        if b in nicht_ram or b in nicht_vram:
            continue
        if summe + b.ausgabe_bytes() > budget.platte:
            break
        summe += b.ausgabe_bytes()
        passen += 1
    # Gleichzeitigkeit fuer die tatsaechliche Mischung: so viele der kleinsten Boxen, wie RAM und VRAM tragen
    machbar = [b for b in offen if b not in nicht_ram and b not in nicht_vram]
    nach_ram = sorted(machbar, key=lambda b: b.ram(1))
    k, ram_rest, vram_rest = 0, budget.ram, budget.vram
    for b in nach_ram:
        if b.ram(1) > ram_rest or b.vram(1) > vram_rest:
            break
        ram_rest -= b.ram(1)
        vram_rest -= b.vram(1)
        k += 1
    gleichzeitig = max(1, min(k, budget.parallel)) if machbar else 0
    gpu = sum(gpu_schaetzung(b, historie)[0] for b in machbar)
    cpu = sum(cpu_schaetzung(b, faeden) for b in machbar)
    dauer = max(gpu, (gpu + cpu) / max(1, gleichzeitig)) if machbar else 0.0
    return dict(offen=len(offen), machbar=len(machbar), nicht_ram=nicht_ram, nicht_vram=nicht_vram,
                platte_passen=passen, platte_summe=summe, ram_k=k, gleichzeitig=gleichzeitig,
                gpu_s=gpu, cpu_s=cpu, dauer_s=dauer, gross_ram=gross_ram)


# --------------------------------------------------------------------------------------------------------------------
# Ausgabe
# --------------------------------------------------------------------------------------------------------------------
def gib(x: int | float) -> str:
    return f"{x / GIB:.1f}".replace(".", ",") + " GiB"


def dauer(s: float) -> str:
    s = max(0.0, s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h".replace(".", ",")


def kurz(text: str, breite: int) -> str:
    return text if len(text) <= breite else "…" + text[-(breite - 1):]


# --------------------------------------------------------------------------------------------------------------------
# Lauf
# --------------------------------------------------------------------------------------------------------------------
class Sensorprotokoll:
    KOPF = "zeit,tj_c,kante_c,speicher_c,leistung_w,takt_mhz,luefter_rpm,cpu_tctl_c,ram_frei_gib,laufend,abkuehlen\n"

    def __init__(self, pfad: Path):
        self.f = pfad.open("a", buffering=1)
        if pfad.stat().st_size == 0:
            self.f.write(self.KOPF)

    def schreiben(self, s: dict, ram_frei: int, laufend: int, abkuehlen: bool) -> None:
        self.f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S},{s['tj']:.0f},{s['kante']:.0f},{s['speicher']:.0f},"
                     f"{s['leistung']:.0f},{s['takt']:.0f},{s['luefter']:.0f},{cpu_tctl():.0f},"
                     f"{ram_frei / GIB:.1f},{laufend},{int(abkuehlen)}\n")
        self.f.flush()
        os.fsync(self.f.fileno())           # nach einem Absturz soll die letzte Zeile auf der Platte stehen

    def schliessen(self) -> None:
        self.f.close()


@dataclass
class Lauf:
    box: Box
    proc: subprocess.Popen
    start: float
    vram: int
    b: int
    log: Path
    rss_max: int = 0
    geteilt: bool = False       # lief zeitweise neben einer anderen Box (GPU-Zeit dann nicht repraesentativ)
    beendet_weil: str = ""      # vom Laeufer beendet (RAM knapp, Waechter)
    waechter: bool = False      # vom Waechter beendet (Box kommt einmal erneut dran)
    tj_max: float = -1.0        # Messungen des Laeufers waehrend der Box (alle 0,5 s)
    kante_max: float = -1.0
    leistung_summe: float = 0.0
    messungen: int = 0


def systemd_scope_pruefen() -> str:
    """Leer, wenn systemd-run --user --scope den Speicherdeckel wirklich setzt; sonst der Grund."""
    if not shutil.which("systemd-run"):
        return "systemd-run fehlt"
    probe = 300 << 20
    try:
        r = subprocess.run(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={probe}", "--", "sh", "-c",
                            'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max"'],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"systemd-run --user --scope geht nicht ({e})"
    if r.returncode != 0:
        return "systemd-run --user --scope geht nicht: " + (r.stderr.strip()[:120] or f"rc {r.returncode}")
    if r.stdout.strip() != str(probe):
        return f"Speicherdeckel greift nicht (memory.max = {r.stdout.strip() or '?'})"
    return ""


def rss(pid: int) -> tuple[int, int]:
    """(VmRSS, VmHWM) in Bytes."""
    werte = {}
    for zeile in _lies(f"/proc/{pid}/status").splitlines():
        if zeile.startswith(("VmRSS:", "VmHWM:")):
            werte[zeile.split(":")[0]] = int(zeile.split()[1]) * 1024
    return werte.get("VmRSS", 0), werte.get("VmHWM", 0)


AUSGABE_ENDUNGEN = (".bts", ".wnd", ".twr", ".hh", ".bin", ".dat", ".u", ".v", ".w", ".cts", ".gpu.ok")


def box_vorbereiten(box: Box) -> None:
    """Eingabe und Nutzerdateien in den Laufordner (immer frisch kopiert), alte Ausgaben dieser Box loeschen."""
    box.laufdir.mkdir(parents=True, exist_ok=True)
    refs = bezogene_dateien(box.inp)
    geschuetzt = {(box.laufdir / box.inp.name).resolve()} | {(box.laufdir / rel).resolve() for rel, _ in refs}
    ziel_inp = box.laufdir / box.inp.name
    if ziel_inp.resolve() != box.inp.resolve():
        shutil.copy2(box.inp, ziel_inp)
        for rel, q in refs:
            z = box.laufdir / rel
            z.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(q, z)                          # klein; immer kopieren (gleiche Groesse heisst nicht gleicher Inhalt)
    for endung in AUSGABE_ENDUNGEN:
        p = box.laufdir / f"{box.stamm}{endung}"
        if p.resolve() not in geschuetzt:               # nie eine Eingabe loeschen, die zufaellig so heisst
            p.unlink(missing_ok=True)


def _kind_an_eltern_binden() -> None:
    """Im Kindprozess vor exec: stirbt der Laeufer (auch SIGKILL, Terminal zu), stirbt die Box mit (PR_SET_PDEATHSIG)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL))
    except (OSError, AttributeError):
        pass


def box_starten(box: Box, b: int, vram: int, prof: dict, args, scope: bool, ram_verfuegbar: int) -> Lauf:
    box_vorbereiten(box)
    env = dict(os.environ)
    for k in ("TS_GPU_TJ_MAX", "TS_GPU_TJ_WIEDER", "TS_GPU_STAPEL_S", "TS_GPU_PAUSE", "TS_GPU_CU", "TS_GPU_SKIP",
              "TS_GPU_TJ_WARTE_MAX_S", "TS_GPU_KANTE_MAX", "TS_GPU_KANTE_WIEDER", "TS_PLAN", "TS_DUMP_V",
              "TS_CHOL_TESTFAIL", "TS_CHOL_SKIP", *GPU_SICHT_VARIABLEN):
        env.pop(k, None)                                # Pruef-/Planschalter und GPU-Umdeutungen aus der Shell nie in Laeufe
    # VRAM-Deckel mit Spielraum ueber dem Modell (echtes K/m weicht ab); die Stapelgroesse begrenzt TS_GPU_B = b.
    kappe = vram - VRAM_KONTEXT + max(64 << 20, box.vram_je_gruppe() // 4)
    env.update(TS_GPU="nur", TS_GPU_INFO="1", TS_ZEITEN="1", TS_CHOL_ISA="avx2",
               OMP_NUM_THREADS=str(args.cpu_faeden), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TS_GPU_B=str(max(1, b)), TS_GPU_VRAM_MAX_GB=f"{math.ceil(kappe / GIB * 1000) / 1000:.3f}",
               TS_GPU_VRAM_RESERVE_GB=f"{args.vram_reserve_gb:g}",
               TS_GPU_TJ_MAX=f"{prof['tj_max']:g}", TS_GPU_TJ_WIEDER=f"{prof['tj_wieder']:g}",
               TS_GPU_KANTE_MAX=f"{prof['kante_max']:g}", TS_GPU_KANTE_WIEDER=f"{prof['kante_wieder']:g}",
               TS_GPU_STAPEL_S=f"{prof['stapel_s']:g}")
    kmd = [str(args.binary), box.inp.name]
    if scope:
        # Deckel ueber der Schaetzung (Spielraum), aber nie ueber den freien RAM hinaus: sonst traefe der
        # systemweite OOM-Killer zuerst andere Programme (Desktop), statt nur diese Box zu beenden.
        deckel = max(box.ram(b), min(int(box.ram(b) * 1.15) + (256 << 20), ram_verfuegbar - (3 << 29)))
        kmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={deckel}", "-p", "MemorySwapMax=0", "--"] + kmd
    log = box.laufdir / f"{box.stamm}.gpu.log"
    with log.open("w") as f:
        f.write(f"# {dt.datetime.now():%F %T} {' '.join(kmd)}\n# B={b} VRAM-Anteil {gib(vram)} RAM-Schaetzung "
                f"{gib(box.ram(b))}\n")
        f.flush()
        proc = subprocess.Popen(kmd, cwd=box.laufdir, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=_kind_an_eltern_binden)
    box.zustand = "laeuft"
    return Lauf(box, proc, time.time(), vram, b, log)


def log_auswerten(log: Path) -> dict:
    text = _lies(log)
    e = {}
    m = re.findall(r"TurbSim-GPU: N = .*-> ([\d.]+) s", text)
    if m:
        e["gpu_schleife_s"] = sum(float(x) for x in m)
    m = re.findall(r"GPU-Zeit ([\d.]+) s", text)
    if m:
        e["gpu_rechen_s"] = sum(float(x) for x in m)            # reine GPU-Zeit (Ereignisse), ohne Bremse und Halt
    m = re.findall(r"Waermebremse .*?: (\d+) Pausen, ([\d.]+) s gewartet", text)
    if m:
        e["bremspausen"] = sum(int(a) for a, _ in m)
        e["gebremst_s"] = sum(float(b) for _, b in m)
    if "Waermebremse ohne Sensor" in text:
        e["ohne_sensor"] = True
    m = re.findall(r"TurbSim-GPU: (.*(?:abgelehnt|HIP-Fehler|kein gfx1201|VRAM reicht nicht|nicht abgekuehlt|Zeitlimit).*)", text)
    if m:
        e["gpu_meldung"] = m[-1]
    e["gerechnet"] = "Processing complete." in text
    e["gpu_gerechnet"] = "Frequenzschleife auf der GPU gerechnet" in text
    fehler = [z.strip() for z in text.splitlines() if "ERROR" in z or "FATAL" in z.upper() or "Aborting" in z]
    if fehler:
        e["fehler"] = " | ".join(fehler[-3:])
    return e


def lauf_abschliessen(l: Lauf, historie: dict) -> None:
    box, rc = l.box, l.proc.returncode
    wand = time.time() - l.start
    e = log_auswerten(l.log)
    e.update(rc=rc, wand_s=wand, rss_max=l.rss_max, b=l.b, geteilt=l.geteilt, waechter=l.waechter)
    if l.messungen:
        e.update(tj_max=l.tj_max, kante_max=l.kante_max, leistung_mittel=l.leistung_summe / l.messungen)
    bts = box.laufdir / f"{box.stamm}.bts"
    groesse = bts.stat().st_size if bts.exists() else -1
    soll = box.bts_bytes()
    # Erfolg nur mit Beleg, dass gerechnet wurde (ein uebriggebliebenes TS_PLAN endete auch mit rc 0)
    if (rc == 0 and e["gerechnet"] and (e["gpu_gerechnet"] or not box.gpu_aufrufe)
            and (soll == 0 or groesse == soll) and not l.beendet_weil):
        box.zustand = "fertig"
        try:
            (box.laufdir / f"{box.stamm}.gpu.ok").write_text(json.dumps(
                {"eingabe": eingabe_hash(box), "bts": soll, "zeit": f"{dt.datetime.now():%F %T}"}) + "\n")
        except (OSError, ValueError):
            pass
        gpu_s = e.get("gpu_rechen_s", 0.0)
        alt = historie.get(box.signatur())
        if gpu_s > 0 and (not l.geteilt or not alt or alt.get("geteilt", False)):
            if alt and alt.get("geteilt", False) and not l.geteilt:
                del historie[box.signatur()]                 # Messung allein ersetzt die geteilte
            h = historie.setdefault(box.signatur(), {"gpu_s": 0.0, "wand_s": 0.0, "n": 0, "geteilt": l.geteilt})
            k = h["n"]
            h["gpu_s"] = (h["gpu_s"] * k + gpu_s) / (k + 1)
            h["wand_s"] = (h["wand_s"] * k + wand) / (k + 1)
            h["n"] = k + 1
    else:
        box.zustand = "gescheitert"
        if l.beendet_weil:
            e["fehler"] = l.beendet_weil
        elif rc in (-signal.SIGKILL, 137):
            e.setdefault("fehler", "vom Speicherdeckel beendet (MemoryMax) - RAM-Schaetzung zu klein?")
        elif rc == 0 and not e["gerechnet"]:
            e.setdefault("fehler", "TurbSim endete ohne Rechnung (TS_PLAN in der Umgebung?)")
        elif rc == 0:
            e.setdefault("fehler", f".bts hat {groesse} statt {soll} Bytes")
        bts.unlink(missing_ok=True)       # halbe Box nie liegen lassen
        (box.laufdir / f"{box.stamm}.gpu.ok").unlink(missing_ok=True)
    box.ergebnis = e


def signal_an(laeufe: list[Lauf], sig: int) -> None:
    for l in laeufe:
        try:
            os.kill(l.proc.pid, sig)
        except ProcessLookupError:
            pass


def ausfuehren(boxen: list[Box], budget: Budget, gpu: Gpu, prof: dict, args, historie: dict, ziel: Path,
               stempel: str, scope: bool) -> int:
    sensor = Sensorprotokoll(ziel / f"sensoren_{stempel}.csv")
    vorrat = sorted([b for b in boxen if b.zustand == "offen"], key=lambda b: -b.ram())
    warteschlange = list(vorrat)
    ende_dauertest = time.time() + 60 * args.dauertest if args.dauertest else 0.0
    laeufe: list[Lauf] = []
    protokoll: list[tuple[Box, str, dict]] = []      # jeder beendete Lauf (Dauertest: jede Wiederholung)
    z = dict(fertig_n=0, gesamt=len(warteschlange), fehler_in_folge=0)
    abkuehlen, abkuehlen_seit = False, 0.0            # nach einer Notabschaltung: erst abkuehlen
    ueber, notfaelle = 0, 0
    nochmal: dict[int, int] = {}                      # id(Box) -> Wiederholungen nach Notabschaltung
    sperre = ""                                       # keine neuen Starts mehr (GPU-Fehler, Fehlerserie, Dauertest-Fehler)
    fremd: list[str] = []
    t_start = time.time()
    letzter_sensor = letzte_pruefung = 0.0
    abbruch = False

    def ende(_sig, _frame):
        nonlocal abbruch
        abbruch = True

    def melden(l: Lauf) -> None:
        nonlocal sperre
        lauf_abschliessen(l, historie)
        speichere_historie(historie)
        z["fertig_n"] += 1
        e = l.box.ergebnis
        protokoll.append((l.box, l.box.zustand, dict(e)))
        kopf = f"  [{z['fertig_n']}/{z['gesamt']}] {kurz(l.box.name, 40)}"
        if l.box.zustand != "fertig" and l.waechter and nochmal.get(id(l.box), 0) < 1 and not sperre:
            nochmal[id(l.box)] = nochmal.get(id(l.box), 0) + 1
            l.box.zustand, l.box.ergebnis = "offen", {}
            warteschlange.insert(0, l.box)                # gleich nach dem Abkuehlen noch einmal
            z["gesamt"] += 1
            print(f"{kopf} vom Waechter beendet - kommt nach dem Abkuehlen noch einmal dran", flush=True)
            return
        if l.box.zustand == "fertig":
            z["fehler_in_folge"] = 0
            zusatz = f"GPU {dauer(e.get('gpu_rechen_s', e.get('gpu_schleife_s', 0)))}"
            if "tj_max" in e:
                zusatz += (f", Tj max {e['tj_max']:.0f} C, Kante max {e['kante_max']:.0f} C, {e['leistung_mittel']:.0f} W "
                           f"im Mittel, gebremst {dauer(e.get('gebremst_s', 0))}")
            if e.get("ohne_sensor"):
                zusatz += ", OHNE Tj-Sensor (Ersatzpausen)"
            print(f"{kopf} fertig in {dauer(e['wand_s'])} ({zusatz}, RSS {gib(e['rss_max'])})", flush=True)
            if args.dauertest:
                (l.box.laufdir / f"{l.box.stamm}.bts").unlink(missing_ok=True)
                (l.box.laufdir / f"{l.box.stamm}.gpu.ok").unlink(missing_ok=True)
            return
        z["fehler_in_folge"] += 1
        grund = e.get("gpu_meldung") or e.get("fehler") or "siehe " + str(l.log)
        print(f"{kopf} GESCHEITERT (rc {e['rc']}): {grund}", flush=True)
        meldung = e.get("gpu_meldung", "")
        if "HIP-Fehler" in meldung or "kein gfx1201" in meldung:
            sperre = f"GPU-Fehler bei {l.box.name} ({meldung})"
        elif l.waechter:
            z["fehler_in_folge"] -= 1                     # Waechter-Abbruch zaehlt eigens (NOTFAELLE_MAX)
        elif "Zeitlimit der Waermebremse" in meldung:
            sperre = f"die Karte kuehlt nicht ab (fremde Last?) bei {l.box.name}"
        elif z["fehler_in_folge"] >= 2:
            sperre = "zwei Boxen nacheinander gescheitert"
        elif args.dauertest:
            sperre = "Dauertest: erster Fehler"

    alte = {sig: signal.signal(sig, ende) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)}
    print(f"\nLauf  (Sensorprotokoll {ziel / f'sensoren_{stempel}.csv'}; Strg+C bricht sauber ab)")
    try:
        while not abbruch and (warteschlange or laeufe):
            jetzt = time.time()
            # --- fertige Laeufe einsammeln ---
            for l in list(laeufe):
                if l.proc.poll() is None:
                    l.rss_max = max(l.rss_max, rss(l.proc.pid)[1])
                    continue
                laeufe.remove(l)
                melden(l)
            if sperre and warteschlange:
                print(f"  HALT: {sperre} - keine weiteren Boxen.", flush=True)
                for b in warteschlange:
                    b.zustand, b.ergebnis = "uebersprungen", {"fehler": f"nicht gestartet: {sperre}"}
                warteschlange = []
            # --- Dauertest: auffuellen, nur mit Boxen, die gerade nicht laufen ---
            if args.dauertest:
                if jetzt >= ende_dauertest:
                    warteschlange = []
                elif not warteschlange and not sperre:
                    laufend = {id(l.box) for l in laeufe}
                    neu = [b for b in vorrat if id(b) not in laufend and b.zustand != "uebersprungen"]
                    for b in neu:
                        b.zustand = "offen"
                    warteschlange = neu
                    z["gesamt"] += len(neu)
            if len(laeufe) > 1:
                for l in laeufe:
                    l.geteilt = True

            # --- Sensoren, Waechter ---
            s = gpu.sensoren()
            mi = meminfo()
            tj = s["tj"]
            if jetzt - letzter_sensor >= 1.0:
                sensor.schreiben(s, mi.get("MemAvailable", 0), len(laeufe), abkuehlen)
                letzter_sensor = jetzt
            for l in laeufe:
                if tj >= 0:
                    l.tj_max = max(l.tj_max, tj)
                    l.kante_max = max(l.kante_max, s["kante"])
                    l.leistung_summe += max(0.0, s["leistung"])
                    l.messungen += 1
            # Waechter: zweimal in Folge ueber der Notfallgrenze -> alle Boxen beenden (SIGKILL stoppt die GPU sofort)
            ueber = ueber + 1 if (laeufe and tj >= prof["notfall"]) else 0
            if ueber >= NOTFALL_MESSUNGEN:
                notfaelle += 1
                for l in laeufe:
                    l.waechter = True
                    l.beendet_weil = f"vom Waechter beendet: Tj {tj:.0f} C >= {prof['notfall']:.0f} C"
                signal_an(laeufe, signal.SIGKILL)
                abkuehlen, abkuehlen_seit, ueber = True, jetzt, 0
                print(f"  WAECHTER: Tj {tj:.0f} C >= {prof['notfall']:.0f} C (Kante {s['kante']:.0f} C, {s['leistung']:.0f} W) - "
                      f"Boxen beendet, Karte kuehlt ab ({notfaelle}. Notabschaltung)", flush=True)
                if notfaelle >= NOTFAELLE_MAX and not sperre:
                    sperre = (f"{notfaelle} Notabschaltungen - die Karte wird trotz Bremse zu heiss (Leistungsgrenze senken, "
                              f"Luefterkurve steiler)")
            elif abkuehlen and not laeufe:
                # weiter erst mit gueltigen Messwerten unter beiden Wiederanlaufgrenzen; ein Sensor ohne Messwert gilt
                # erst nach 60 s Ruhe als kuehl (die Boxen sind beendet, die GPU ist leer)
                lange = jetzt - abkuehlen_seit >= 60.0
                tj_ok = 0 <= tj <= prof["tj_wieder"] or (tj < 0 and lange)
                kante_ok = 0 <= s["kante"] <= prof["kante_wieder"] or (s["kante"] < 0 and lange)
                if tj_ok and kante_ok:
                    abkuehlen = False
                    print(f"  WAECHTER: abgekuehlt (Tj {tj:.0f} C, Kante {s['kante']:.0f} C) - weiter", flush=True)
            # RAM-Notbremse: lieber die juengste Box opfern als den Rechner
            if laeufe and mi.get("MemAvailable", GIB * 1024) < int(args.ram_not_gb * GIB):
                l = max(laeufe, key=lambda x: x.start)
                if not l.beendet_weil:
                    l.beendet_weil = (f"vom Laeufer beendet: nur noch {gib(mi.get('MemAvailable', 0))} RAM frei "
                                      f"(Notgrenze {args.ram_not_gb:g} GiB)")
                    print(f"  RAM-NOTBREMSE: {kurz(l.box.name, 40)} beendet ({gib(mi.get('MemAvailable', 0))} frei)", flush=True)
                    signal_an([l], signal.SIGKILL)
            # fremde GPU-Last waehrend des Laufs: keine neuen Boxen, bis sie weg ist
            if jetzt - letzte_pruefung >= 10.0:
                letzte_pruefung = jetzt
                neu_fremd = fremde_gpu_nutzer(gpu.pci, ausser={l.proc.pid for l in laeufe})
                if neu_fremd and not fremd:
                    print("  WARNUNG: fremde GPU-Last (" + ", ".join(neu_fremd) + ") - keine neuen Boxen, bis sie weg ist",
                          flush=True)
                elif fremd and not neu_fremd:
                    print("  fremde GPU-Last beendet - weiter", flush=True)
                fremd = neu_fremd

            # --- naechste Box starten (hoechstens eine je Takt) ---
            if not abkuehlen and not sperre and not fremd and warteschlange and len(laeufe) < budget.parallel:
                wachsend = sum(max(0, l.box.ram(l.b) - rss(l.proc.pid)[0]) for l in laeufe)   # noch wachsende Boxen
                verfuegbar = mi.get("MemAvailable", 0) - wachsend
                ram_frei = verfuegbar - int(args.ram_reserve_gb * GIB)
                vram_frei = budget.vram - sum(l.vram for l in laeufe)
                platte_frei = shutil.disk_usage(ziel).free - int(args.platte_reserve_gb * GIB)
                platte_frei -= sum(l.box.ausgabe_bytes() for l in laeufe)
                for box in list(warteschlange):
                    b, anteil = waehle_b(box, vram_frei, ram_frei)
                    if b > 0 and box.ausgabe_bytes() <= platte_frei:
                        warteschlange.remove(box)
                        try:
                            laeufe.append(box_starten(box, b, anteil, prof, args, scope, verfuegbar))
                        except (OSError, ValueError) as e:
                            box.zustand, box.ergebnis = "gescheitert", {"fehler": f"Start fehlgeschlagen: {e}"}
                            z["fertig_n"] += 1
                            protokoll.append((box, box.zustand, dict(box.ergebnis)))
                            print(f"  [{z['fertig_n']}/{z['gesamt']}] {kurz(box.name, 40)} {box.ergebnis['fehler']}", flush=True)
                            break
                        print(f"  start {kurz(box.name, 40)}: B={b}, VRAM {gib(anteil)}, RAM {gib(box.ram(b))}"
                              f" ({len(laeufe)} laufen)", flush=True)
                        break
                    if not laeufe:                      # allein passt sie nicht: nie startbar
                        warteschlange.remove(box)
                        box.zustand = "uebersprungen"
                        box.ergebnis = {"fehler": f"passt nicht (RAM frei {gib(ram_frei)}, Bedarf {gib(box.ram(1))}; VRAM "
                                                  f"frei {gib(vram_frei)}, Bedarf {gib(box.vram(1))}; Platte frei "
                                                  f"{gib(platte_frei)}, Bedarf {gib(box.ausgabe_bytes())})"}
                        z["fertig_n"] += 1
                        protokoll.append((box, box.zustand, dict(box.ergebnis)))
                        print(f"  [{z['fertig_n']}/{z['gesamt']}] {kurz(box.name, 40)} uebersprungen: "
                              f"{box.ergebnis['fehler']}", flush=True)
            time.sleep(0.5)
    finally:
        for l in list(laeufe):                          # in der letzten Runde fertig geworden: normal auswerten
            if l.proc.poll() is not None:
                laeufe.remove(l)
                melden(l)
        if laeufe:
            print("  Abbruch: laufende Boxen werden beendet und ihre halben Ausgaben geloescht.")
            signal_an(laeufe, signal.SIGCONT)
            signal_an(laeufe, signal.SIGTERM)
            for l in laeufe:
                try:
                    l.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    l.proc.kill()
                    l.proc.wait()
                l.box.zustand = "abgebrochen"
                l.box.ergebnis = {"fehler": "Lauf abgebrochen"}
                protokoll.append((l.box, "abgebrochen", dict(l.box.ergebnis)))
                (l.box.laufdir / f"{l.box.stamm}.bts").unlink(missing_ok=True)
                (l.box.laufdir / f"{l.box.stamm}.gpu.ok").unlink(missing_ok=True)
        sensor.schliessen()
        for sig, alt in alte.items():
            signal.signal(sig, alt)
    bericht(boxen, protokoll, ziel / f"bericht_{stempel}.txt", time.time() - t_start, bool(args.dauertest))
    if args.dauertest:
        return 0 if protokoll and not abbruch and all(zu == "fertig" for _, zu, _ in protokoll) else 1
    return 0 if all(b.zustand in ("fertig", "fertig_vorhanden") for b in boxen) else 1


def bericht(boxen: list[Box], protokoll: list, pfad: Path, wand: float, dauertest: bool) -> None:
    zeilen = [f"gpu_windfelder.py {dt.datetime.now():%F %T}, Wanduhr {dauer(wand)}", ""]

    def info(zustand: str, e: dict) -> str:
        if zustand == "fertig":
            return (f"{dauer(e.get('wand_s', 0))}, GPU {dauer(e.get('gpu_rechen_s', e.get('gpu_schleife_s', 0)))}, "
                    f"RSS {gib(e.get('rss_max', 0))}"
                    + (f", Tj max {e['tj_max']:.0f} C, gebremst {dauer(e.get('gebremst_s', 0))}" if "tj_max" in e else ""))
        return (e.get("gpu_meldung") or e.get("fehler") or "") if e else ""

    zaehler: dict[str, int] = {}
    if dauertest:                                       # jede Wiederholung zaehlt
        for b, zustand, e in protokoll:
            zaehler[zustand] = zaehler.get(zustand, 0) + 1
            zeilen.append(f"{zustand:16s} {b.laufdir / b.stamm}  {info(zustand, e)}")
    else:
        for b in boxen:
            zaehler[b.zustand] = zaehler.get(b.zustand, 0) + 1
            zeilen.append(f"{b.zustand:16s} {b.laufdir / b.stamm}  {info(b.zustand, b.ergebnis)}")
    zeilen.append("")
    zeilen.append("  ".join(f"{k}: {v}" for k, v in sorted(zaehler.items())))
    try:
        pfad.write_text("\n".join(zeilen) + "\n")
    except OSError:
        pass
    print("\nErgebnis" + (" (alle Wiederholungen)" if dauertest else "") + ": "
          + "  ".join(f"{k} {v}" for k, v in sorted(zaehler.items())) + f"  (Bericht {pfad})")


# --------------------------------------------------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------------------------------------------------
def binary_pruefen(binary: Path) -> str:
    if not binary.is_file():
        return (f"Binary fehlt: {binary} (bauen: GPU=ON ./turbsim/build_turbsim_ref353.sh <openfast-v3.5.3> build/turbsim-gpu, "
                f"oder --binary / TURBSIM_GPU_BIN setzen)")
    inhalt = binary.read_bytes()
    for marke in (b"tsg_pflicht", b"TS_PLAN", b"TS_GPU_TJ_MAX"):
        if marke not in inhalt:
            return f"{binary} ist zu alt (ohne {marke.decode()}): mit GPU=ON neu bauen"
    return ""


def laufdir_fuer(inp: Path, wurzel: Path, ziel: Path) -> Path:
    rel = os.path.relpath(inp.parent, wurzel)
    return (ziel / rel).resolve() if rel != "." else ziel.resolve()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TurbSim-Windfelder nur auf der GPU, mit RAM/VRAM/Platten-Planung.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Beispiele", 1)[1])
    ap.add_argument("eingaben", nargs="+", help=".inp-Dateien oder Ordner (rekursiv, _templates ausgenommen)")
    ap.add_argument("--ziel", type=Path, default=Path("windfelder_gpu"),
                    help="Ausgabeordner; die Ordnerstruktur der Eingaben wird gespiegelt (Vorgabe ./windfelder_gpu)")
    ap.add_argument("--profil", choices=sorted(PROFILE), default="sicher",
                    help="Waermebremse nach der Kantentemperatur: sicher 58/53 C (Leistungsgrenze <= 215 W), normal 62/57, "
                         "voll 66/61; Waechter beendet bei Tj 93/95/97 C; Vorgabe sicher")
    ap.add_argument("--kante", type=float, default=0.0, metavar="C",
                    help="Waermebremse ab dieser Kantentemperatur (weiter 5 C darunter), statt des Profilwerts")
    ap.add_argument("--notfall", type=float, default=0.0, metavar="C",
                    help="Waechter beendet die Boxen ab dieser Sperrschichttemperatur, statt des Profilwerts")
    ap.add_argument("--parallel", type=int, default=2,
                    help="hoechstens so viele Boxen gleichzeitig (Vorgabe 2: die GPU rechnet, die zweite Box ueberdeckt "
                         "Einlesen/FFT/Ausgabe der ersten; mehr bringt nichts)")
    ap.add_argument("--nur-planen", action="store_true", help="nur System erfassen und planen, nichts rechnen")
    ap.add_argument("--ram-reserve-gb", type=float, default=4.0, help="RAM, der fuer System/Desktop frei bleibt (4)")
    ap.add_argument("--vram-reserve-gb", type=float, default=3.0, help="VRAM, das frei bleibt (3)")
    ap.add_argument("--platte-reserve-gb", type=float, default=10.0, help="Plattenplatz, der frei bleibt (10)")
    ap.add_argument("--ram-not-gb", type=float, default=1.5,
                    help="faellt der freie RAM waehrend des Laufs darunter, wird die juengste Box beendet (1,5)")
    ap.add_argument("--cpu-faeden", type=int, default=2, help="OpenMP-Faeden je Box fuer FFT/Ausgabe (2)")
    ap.add_argument("--binary", type=Path, default=Path(os.environ.get("TURBSIM_GPU_BIN", BIN_VORGABE)))
    ap.add_argument("--neu", action="store_true", help="auch Boxen rechnen, deren .bts schon vollstaendig da ist")
    ap.add_argument("--trotz-warnung", action="store_true",
                    help="trotz Warnungen rechnen (fremde GPU-Last, laufende Rechnungen, Leistungsgrenze > 255 W oder "
                         "unlesbar, kein Tj-Sensor, Speicherdeckel greift nicht)")
    ap.add_argument("--dauertest", type=float, default=0.0, metavar="MIN",
                    help="Stabilitaetstest: die Eingaben MIN Minuten lang wiederholt rechnen, in einem frischen Unterordner "
                         "<ziel>/dauertest_<zeit> (die erzeugten .bts werden dort geloescht); Halt beim ersten Fehler")
    args = ap.parse_args(argv)

    fehler = binary_pruefen(args.binary)
    if fehler:
        print("FEHLER: " + fehler, file=sys.stderr)
        return 2
    stempel = f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    eingaben = finde_eingaben(args.eingaben, args.ziel)
    if not eingaben:
        print("FEHLER: keine TurbSim-Eingaben gefunden.", file=sys.stderr)
        return 2
    gpu = finde_gpu()
    if gpu is None:
        print("FEHLER: keine AMD-GPU mit >= 4 GB VRAM gefunden - nur-GPU-Betrieb unmoeglich.", file=sys.stderr)
        return 2
    prof = dict(PROFILE[args.profil])
    if args.notfall:
        prof["notfall"] = args.notfall
    if args.kante:
        prof["kante_max"], prof["kante_wieder"] = args.kante, args.kante - 5.0
    ziel = args.ziel.resolve()
    if args.dauertest:                       # frischer Ordner: nie vorhandene Felder loeschen
        ziel = ziel / f"dauertest_{stempel}"
    ziel.mkdir(parents=True, exist_ok=True)
    wurzel = Path(os.path.commonpath([str(p.parent) for p in eingaben]))

    # --- 1. System ---
    mi = meminfo()
    fd = gpu.wecken()                        # aus dem Runtime-Suspend holen, sonst sind Sensoren/Leistungsgrenze unlesbar
    s = gpu.sensoren()
    ram_nutzbar = mi.get("MemAvailable", 0) - int(args.ram_reserve_gb * GIB)
    vram_fremd = gpu.vram_belegt()
    vram_nutzbar = gpu.vram_ges - int(args.vram_reserve_gb * GIB) - vram_fremd
    platte = shutil.disk_usage(ziel)
    platte_nutzbar = platte.free - int(args.platte_reserve_gb * GIB)
    grenze = gpu.leistungsgrenze_w()
    if fd >= 0:
        os.close(fd)
    print("System")
    print(f"  RAM      {gib(mi.get('MemTotal', 0))} gesamt, {gib(mi.get('MemAvailable', 0))} verfuegbar, Reserve "
          f"{args.ram_reserve_gb:g} GiB -> {gib(max(0, ram_nutzbar))} fuer TurbSim; Swap {gib(mi.get('SwapFree', 0))} frei "
          f"(wird je Box gesperrt)")
    print(f"  GPU      {gpu.pci}: VRAM {gib(gpu.vram_ges)}, belegt {gib(vram_fremd)}, Reserve {args.vram_reserve_gb:g} GiB "
          f"-> {gib(max(0, vram_nutzbar))} fuer TurbSim")
    print(f"           Leistungsgrenze {grenze:.0f} W, Tj {s['tj']:.0f} C, Kante {s['kante']:.0f} C, {s['leistung']:.0f} W, "
          f"Luefter {s['luefter']:.0f} rpm")
    print(f"  CPU      {cpu_name()}, {os.cpu_count()} Faeden; je Box {args.cpu_faeden} Faeden nur fuer Einlesen/FFT/Ausgabe, "
          f"Tctl {cpu_tctl():.0f} C")
    print(f"  Platte   {ziel}: {gib(platte.free)} frei, Reserve {args.platte_reserve_gb:g} GiB")
    print(f"  Profil   {args.profil}: Waermebremse Kante {prof['kante_max']:.0f}/{prof['kante_wieder']:.0f} C und Tj "
          f"{prof['tj_max']:.0f}/{prof['tj_wieder']:.0f} C, Stapel ~{prof['stapel_s']:g} s; Waechter beendet die Boxen bei Tj "
          f"{prof['notfall']:.0f} C" + (f"; Leistungsgrenze hoechstens {prof['leistung_max']:.0f} W" if prof["leistung_max"] else ""))
    warnungen: list[tuple[str, str]] = []    # (Text, "hart" = nie rechnen | "weich" = nur mit --trotz-warnung)
    if prof["leistung_max"] and grenze > prof["leistung_max"] + 0.5:
        warnungen.append((f"Leistungsgrenze {grenze:.0f} W > {prof['leistung_max']:.0f} W fuer Profil {args.profil} "
                          f"(in LACT senken oder anderes Profil)", "weich"))
    elif grenze > LEISTUNG_GETESTET_W + 1:
        warnungen.append((f"Leistungsgrenze {grenze:.0f} W > {LEISTUNG_GETESTET_W} W: bei 300 W hing der Rechner am 14.10. "
                          f"nach Minuten (in LACT senken)", "weich"))
    elif grenze < 0:
        warnungen.append(("Leistungsgrenze nicht lesbar (Karte schlaeft oder hwmon fehlt) - 255 W nicht geprueft", "weich"))
    if gpu.tj_pfad is None or s["tj"] < 0:
        warnungen.append(("kein Tj-Sensor lesbar: Waermebremse nur im Ersatzbetrieb (50 % Pause), Waechter blind", "weich"))
    fremd = fremde_gpu_nutzer(gpu.pci)
    if fremd:
        warnungen.append(("andere Prozesse auf der GPU: " + ", ".join(fremd), "weich"))
    rechner = laufende_rechner()
    if rechner:
        warnungen.append(("laufende Rechnungen (ihr RAM-Wachstum ist nicht eingeplant): " + ", ".join(rechner[:6]), "weich"))
    scope_grund = systemd_scope_pruefen()
    if scope_grund:
        warnungen.append((f"{scope_grund} - ohne Speicherdeckel je Box", "weich"))
    if ram_nutzbar <= 0 or vram_nutzbar <= 0:
        warnungen.append(("kein nutzbarer RAM/VRAM nach Reserve", "hart"))
    gesetzt = [f"{k}={os.environ[k]}" for k in GPU_SICHT_VARIABLEN if k in os.environ]
    if gesetzt:
        print("  Hinweis  " + ", ".join(gesetzt) + " aus der Shell wird fuer TurbSim entfernt (Karte ist gfx1201)")
    for w, art in warnungen:
        print("  WARNUNG: " + w)

    # --- 2. Boxen vermessen ---
    historie = lade_historie()
    boxen: list[Box] = []
    arbeit = Path(tempfile.mkdtemp(prefix="gpu_windfelder_"))
    try:
        for inp in eingaben:
            laufdir = laufdir_fuer(inp, wurzel, ziel)
            text = inp.read_text(errors="replace")
            try:
                plan = vermessen(args.binary, inp, arbeit)
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as e:
                b = Box(inp, laufdir, {"npoints": 0, "npacked": 0, "numsteps": 0, "numfreq": 0, "numoutsteps": 0,
                                       "ngrid": 0, "ny": 0, "nz": 0, "ntwr": 0, "usr": 0, "scmod": [0, 0, 0],
                                       "cohgleich": 0, "wr": "000000000", "desclen": 0})
                b.fehler, b.zustand = f"nicht planbar: {e}", "uebersprungen"
                b.ergebnis = {"fehler": b.fehler}
                b.anzeige = str(inp.relative_to(wurzel))
                boxen.append(b)
                continue
            b = Box(inp, laufdir, plan, anzeige=str(inp.relative_to(wurzel)))
            try:
                b.timestep = float(eingabewert(text, "TimeStep") or 0)
                b.uref = float(eingabewert(text, "URef") or 0)
            except ValueError:
                pass
            grund = b.gpu_faehig()
            if grund:
                b.fehler, b.zustand = grund, "uebersprungen"
                b.ergebnis = {"fehler": "nicht nur-GPU-faehig: " + grund}
            elif not args.neu and not args.dauertest and b.fertig_vorhanden():
                b.zustand = "fertig_vorhanden"
            boxen.append(b)
    finally:
        shutil.rmtree(arbeit, ignore_errors=True)

    budget = Budget(ram=max(0, ram_nutzbar), vram=max(0, vram_nutzbar), platte=max(0, platte_nutzbar),
                    parallel=max(1, args.parallel))

    print(f"\nBoxen ({len(boxen)}; Ausgabe gespiegelt nach {ziel})")
    print(f"  {'Box':40s} {'Punkte':>6s} {'Freq.':>6s} {'RAM':>9s} {'VRAM (B)':>13s} {'.bts':>9s} {'gleichz.':>8s} "
          f"{'GPU-Zeit':>12s}  Zustand")
    for b in boxen:
        if b.plan["npoints"] == 0:
            print(f"  {kurz(b.name, 40):40s} {b.ergebnis.get('fehler', '')}")
            continue
        bb, anteil = waehle_b(b, budget.vram, budget.ram)
        k_ram, k_vram = max_gleichzeitig(b, budget)
        g, art = gpu_schaetzung(b, historie)
        zustand = {"offen": "zu rechnen", "fertig_vorhanden": "fertig (.bts vollstaendig)"}.get(b.zustand,
                                                                                               b.ergebnis.get("fehler", b.zustand))
        vtxt = ("—" if not b.gpu_aufrufe else f"{gib(anteil)} ({bb})") if bb else "passt nicht"
        print(f"  {kurz(b.name, 40):40s} {b.n:6d} {b.plan['numfreq']:6d} {gib(b.ram(max(bb, 1))):>9s} "
              f"{vtxt:>13s} {gib(b.bts_bytes()):>9s} {min(k_ram, k_vram):>8d} {dauer(g) + (' ' + art if g else ''):>12s}  "
              f"{zustand}")

    pl = plan_erstellen(boxen, budget, args.cpu_faeden, historie)
    print("\nPlan")
    if pl["offen"] == 0:
        print("  nichts zu rechnen.")
        return 0 if all(b.zustand in ("fertig", "fertig_vorhanden") for b in boxen) else 1
    print(f"  zu rechnen: {pl['offen']} Boxen, davon machbar {pl['machbar']}")
    for b in pl["nicht_ram"]:
        print(f"  passt NICHT in den RAM: {b.stamm} braucht {gib(b.ram(1))}, nutzbar {gib(budget.ram)}")
    for b in pl["nicht_vram"]:
        print(f"  passt NICHT ins VRAM: {b.stamm} braucht {gib(b.vram(1))} (1 Gruppe), nutzbar {gib(budget.vram)}")
    print(f"  gleichzeitig: {pl['gleichzeitig']} (RAM/VRAM tragen {pl['ram_k']} der kleinsten Boxen zugleich, "
          f"Obergrenze --parallel {budget.parallel})")
    print(f"  Platte: {pl['platte_passen']} von {pl['machbar']} Boxen passen ({gib(pl['platte_summe'])} von "
          f"{gib(budget.platte)} nutzbar)")
    print(f"  Dauer: ~{dauer(pl['dauer_s'])} ohne Bremspausen (GPU {dauer(pl['gpu_s'])}, CPU-Anteil {dauer(pl['cpu_s'])}; "
          f"'grob' = +-50 %, wird nach jeder Box genauer)")
    if args.nur_planen:
        return 0
    hart = [w for w, art in warnungen if art == "hart"]
    weich = [w for w, art in warnungen if art == "weich"]
    if hart or (weich and not args.trotz_warnung):
        print("\nAbbruch wegen der Warnungen oben (erst beheben" + ("" if hart else " oder --trotz-warnung") + ").",
              file=sys.stderr)
        return 3
    # nur ein Laeufer zur Zeit: zwei wuessten nichts von RAM und VRAM des anderen
    HISTORIE.parent.mkdir(parents=True, exist_ok=True)
    sperre = (HISTORIE.parent / "lauf.lock").open("w")
    try:
        fcntl.flock(sperre, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"FEHLER: ein anderer gpu_windfelder.py-Lauf ist aktiv ({HISTORIE.parent / 'lauf.lock'}).", file=sys.stderr)
        return 3
    try:
        return ausfuehren(boxen, budget, gpu, prof, args, historie, ziel, stempel, not scope_grund)
    finally:
        sperre.close()


if __name__ == "__main__":
    raise SystemExit(main())

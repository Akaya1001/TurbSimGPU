#!/usr/bin/env python3
"""Prueft zwei TurbSim-Windfelder (.bts) auf Bitgleichheit.

Gegenstueck zu vergleich_outb.py, nur fuer das Windfeld statt der Lastreihen.
Der Massstab ist Bitgleichheit: Wer nur die Reihenfolge der Rechnung aendert
(Parallelisierung der Frequenzschleife), darf kein einziges Datenbyte
verschieben. Weicht etwas ab, ist es ein Fehler und keine Toleranzfrage.

Der Kopf traegt Erzeugungsdatum und -uhrzeit; zwei Laeufe sind dort nie gleich.
Er wird darum feldweise verglichen und die Beschreibung nur bis auf den
Zeitstempel.

Kopfaufbau (TS_FileIO.f90:2380 ff.):
    int16   FileID
    int32   NumGrid_Z, NumGrid_Y, NumTower, NumOutSteps
    float32 dz, dy, dt, UHub, HubHt, Zbottom
    float32 UScl, UOff, VScl, VOff, WScl, WOff
    int32   LenDesc
    char    DescStr(LenDesc)
    int16   Daten ...

Verwendung:  python3 vergleich_bts.py a.bts b.bts [--version-frei]
  --version-frei  maskiert zusaetzlich das Versions-Token (OpenFAST-v3.5.3[+ts2]) in der
                  Beschreibung: Produktionsboxen tragen "+ts2", der Bestand nicht.
Der Datenblock wird per numpy-memmap blockweise verglichen (W960: 2,3 GB je Datei, RAM < 1 GB).
"""
import re
import struct
import sys
from pathlib import Path

import numpy as np

KOPF = struct.Struct("<h4i12f i")   # 2 + 16 + 24 + 24 + 4 = 70 Bytes
FELDER = ("FileID", "NumGrid_Z", "NumGrid_Y", "NumTower", "NumOutSteps",
          "dz", "dy", "dt", "UHub", "HubHt", "Zbottom",
          "UScl", "UOff", "VScl", "VOff", "WScl", "WOff", "LenDesc")
# 'on 29-Aug-2026 at 18:25:27.' — genau dieser Teil darf abweichen.
ZEITSTEMPEL = re.compile(rb" on \d\d-\w\w\w-\d{4} at \d\d:\d\d:\d\d\.")
VERSION = re.compile(rb"OpenFAST-v?\d+\.\d+\.\d+[^ )]*")


def lies(p: Path):
    with open(p, "rb") as f:
        kopf = f.read(KOPF.size)
        werte = KOPF.unpack_from(kopf, 0)
        lendesc = werte[-1]
        desc = f.read(lendesc)
    offset = KOPF.size + lendesc
    n = p.stat().st_size - offset
    daten = np.memmap(p, dtype=np.uint8, mode="r", offset=offset, shape=(n,)) if n > 0 else np.zeros(0, np.uint8)
    return dict(zip(FELDER, werte)), desc, daten


def daten_vergleich(xa, xb, block=1 << 26):
    """(Zahl abweichender Bytes, erster Versatz) — blockweise, ohne beide Dateien zu laden."""
    n = 0; erste = None
    for s in range(0, len(xa), block):
        a = np.asarray(xa[s:s + block]); b = np.asarray(xb[s:s + block])
        d = np.nonzero(a != b)[0]
        if d.size:
            n += int(d.size)
            if erste is None:
                erste = s + int(d[0])
    return n, erste


def main(a: Path, b: Path, version_frei: bool = False) -> int:
    ka, da, xa = lies(a)
    kb, db, xb = lies(b)
    fehler = []

    for f in FELDER:
        if f == "LenDesc":
            continue
        if ka[f] != kb[f]:
            fehler.append(f"Kopf {f}: {ka[f]} != {kb[f]}")

    na = ZEITSTEMPEL.sub(b" on <zeit>.", da)
    nb = ZEITSTEMPEL.sub(b" on <zeit>.", db)
    if version_frei:
        na = VERSION.sub(b"OpenFAST-<version>", na)
        nb = VERSION.sub(b"OpenFAST-<version>", nb)
    if na != nb:
        fehler.append(f"Beschreibung (ohne Zeitstempel):\n  {na!r}\n  {nb!r}")

    if len(xa) != len(xb):
        fehler.append(f"Datenlaenge: {len(xa)} != {len(xb)} Bytes")
    else:
        n, erste = daten_vergleich(xa, xb)
        if n:
            fehler.append(f"Daten: {n} von {len(xa)} Bytes unterschiedlich, "
                          f"erste Abweichung bei Datenversatz {erste}")

    if fehler:
        print(f"NICHT bitgleich: {a.name} gegen {b.name}")
        for f in fehler:
            print("  " + f)
        return 1

    print(f"BITGLEICH: {a.name} gegen {b.name} "
          f"({len(xa)} Datenbytes, {ka['NumGrid_Z']}x{ka['NumGrid_Y']} Punkte, "
          f"{ka['NumOutSteps']} Schritte)")
    return 0


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    sys.exit(main(Path(args[0]), Path(args[1]), "--version-frei" in sys.argv))

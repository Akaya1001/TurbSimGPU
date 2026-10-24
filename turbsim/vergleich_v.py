#!/usr/bin/env python3
"""Vergleicht zwei V-Dumps (double-Fourierkoeffizienten vor der IFFT) bitweise.

Dateiformat (patch_turbsim_dump.py): 3 x int32 (NumSteps, NPoints, 3), dann
V(NumSteps,NPoints,3) als native double in Fortran-Reihenfolge.

Massstab ist Bitgleichheit der Worte. Gemeldet werden: Zahl abweichender
Worte, erste Abweichung als (ITime, IPoint, IVec), groesster ULP-Abstand.
Mit --nullvorzeichen-egal zaehlen +0/-0 nicht als Abweichung (nur fuer
Zwischenfelder sinnvoll; in V darf das nie vorkommen).

Arbeitet mit np.memmap in Bloecken, braucht also auch bei 9-GB-Dumps < 1 GB RAM.

Verwendung:  python3 vergleich_v.py ref.bin neu.bin [--nullvorzeichen-egal]
"""
import sys
import numpy as np


def kopf(p):
    k = np.fromfile(p, dtype=np.int32, count=3)
    return int(k[0]), int(k[1]), int(k[2])


def main(a, b, nullvz_egal=False):
    ka, kb = kopf(a), kopf(b)
    if ka != kb:
        print(f"NICHT vergleichbar: Kopf {ka} != {kb}")
        return 1
    ns, npnt, nc = ka
    n = ns * npnt * nc
    ma = np.memmap(a, dtype=np.uint64, mode="r", offset=12, shape=(n,))
    mb = np.memmap(b, dtype=np.uint64, mode="r", offset=12, shape=(n,))
    block = 1 << 24
    anz = 0
    erste = None
    max_ulp = 0
    nullvz = 0
    for s in range(0, n, block):
        x = np.asarray(ma[s:s + block]); y = np.asarray(mb[s:s + block])
        d = np.nonzero(x != y)[0]
        if d.size == 0:
            continue
        if nullvz_egal:
            fx = x[d].view(np.float64); fy = y[d].view(np.float64)
            beide_null = (fx == 0.0) & (fy == 0.0)
            nullvz += int(beide_null.sum())
            d = d[~beide_null]
            if d.size == 0:
                continue
        anz += d.size
        if erste is None:
            e = int(d[0]) + s
            it = e % ns + 1; ip = (e // ns) % npnt + 1; iv = e // (ns * npnt) + 1
            erste = (it, ip, iv, float(np.asarray(ma[e:e+1]).view(np.float64)[0]), float(np.asarray(mb[e:e+1]).view(np.float64)[0]))
        # ULP-Abstand ueber die Ganzzahl-Darstellung (gleiches Vorzeichen vorausgesetzt)
        xi = x[d].astype(np.int64); yi = y[d].astype(np.int64)
        max_ulp = max(max_ulp, int(np.max(np.abs(xi - yi))))
    if anz == 0:
        extra = f", {nullvz} Nullvorzeichen ignoriert" if nullvz_egal and nullvz else ""
        print(f"BITGLEICH: {n} Worte ({ns} x {npnt} x {nc}){extra}")
        return 0
    print(f"NICHT bitgleich: {anz} von {n} Worten verschieden; erste bei (ITime={erste[0]}, IPoint={erste[1]}, IVec={erste[2]}): "
          f"{erste[3]!r} vs {erste[4]!r}; max ULP-Abstand {max_ulp}")
    return 1


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    sys.exit(main(args[0], args[1], "--nullvorzeichen-egal" in sys.argv))

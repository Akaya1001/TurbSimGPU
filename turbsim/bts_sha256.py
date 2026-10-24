#!/usr/bin/env python3
"""SHA-256 of the data block of a TurbSim .bts file (the header carries a generation timestamp and is skipped).

Two runs are bit-identical exactly when these hashes match. Header layout (TS_FileIO.f90, WrBinTURBSIM): int16 FileID,
4 x int32, 12 x float32, int32 LenDesc at byte 66, then LenDesc characters; the int16 data follow at byte 70 + LenDesc.

Usage:  python3 bts_sha256.py file.bts [more.bts ...]
"""
import hashlib
import struct
import sys
from pathlib import Path


def datenblock_sha256(pfad: Path) -> tuple[str, int]:
    with pfad.open("rb") as f:
        kopf = f.read(70)
        if len(kopf) < 70:
            raise ValueError(f"{pfad}: too short for a .bts header")
        (lendesc,) = struct.unpack_from("<i", kopf, 66)
        f.seek(70 + lendesc)
        h, n = hashlib.sha256(), 0
        while block := f.read(1 << 24):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for a in sys.argv[1:]:
        digest, n = datenblock_sha256(Path(a))
        print(f"{digest}  {n:>12d} data bytes  {a}")

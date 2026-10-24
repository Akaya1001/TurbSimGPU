#!/usr/bin/env python3
"""Kleine Hilfen fuer idempotente Quelltext-Patches an OpenFAST.

Warum eigene Hilfen: Die Fortran-Quellen von OpenFAST wiederholen ganze
Deklarationsbloecke woertlich (CalcFourierCoeffs_IEC und _GENERAL sind bis auf
eine Zeile gleich). Ein globaler Fundstring trifft dann mehrfach und der Patch
landet an der falschen Stelle. Deshalb wird hier immer erst die Routine
ausgeschnitten und nur INNERHALB davon ersetzt.
"""
from __future__ import annotations


def routine_grenzen(s: str, name: str) -> tuple[int, int]:
    """Anfang und Ende einer Routine, die am Zeilenanfang steht."""
    kopf = f"\nSUBROUTINE {name}"
    i = s.find(kopf)
    if i < 0:
        raise AssertionError(f"Routine {name} nicht gefunden")
    if s.find(kopf, i + 1) >= 0:
        raise AssertionError(f"Routine {name} mehrfach gefunden")
    fuss = f"END SUBROUTINE {name}"
    j = s.find(fuss, i)
    if j < 0:
        raise AssertionError(f"Ende von {name} nicht gefunden")
    return i + 1, j + len(fuss)


def ersetze_in_routine(s: str, name: str, alt: str, neu: str) -> str:
    """Ersetzt `alt` genau einmal innerhalb der Routine `name`."""
    a, e = routine_grenzen(s, name)
    koerper = s[a:e]
    n = koerper.count(alt)
    if n != 1:
        raise AssertionError(
            f"{name}: Fundstring {n}x statt 1x vorhanden — Patch pruefen.\n"
            f"  gesucht: {alt[:70]!r}…"
        )
    return s[:a] + koerper.replace(alt, neu, 1) + s[e:]

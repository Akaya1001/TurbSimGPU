#!/usr/bin/env python3
"""Wanduhr je Phase des TurbSim-Hauptprogramms (idempotent, nur TurbSim.f90).

Zweck: den seriellen Rest ausserhalb der Frequenzschleife messen (Phasen, FFT,
Statistik, Mittelwind, .bts), der nach Stufe 1 den Gewinn deckelt (Amdahl).
Reine Zeitstempel (SYSTEM_CLOCK) und eine Tabelle am Ende; keine Numerik.
Ausgabe nur, wenn TS_ZEITEN gesetzt ist.

Verwendung:  python3 patch_turbsim_zeiten.py <baum>/modules/turbsim/src/TurbSim.f90
"""
import sys
from pathlib import Path

MARKE = "tsZeit"

DEKL_ANKER = "CHARACTER(1024)                  :: tsDumpPfad          ! TS_DUMP_V: Zielpfad fuer den V-Dump"
DEKL_NEU = (DEKL_ANKER + "\n"
            "INTEGER(8)                       :: tsZeit(0:12), tsZeitRate   ! Wanduhr je Phase (TS_ZEITEN)\n"
            "CHARACTER(16)                    :: tsZeitAn")

# (Anker, Phasenindex) — der Stempel steht jeweils VOR dem Anker; Phase k endet dort.
STEMPEL = [
    ("CALL CalcTargetPSD(p, S, U, ErrStat, ErrMsg)\n", 1),                        # 0->1: Einlesen, Gitter, Profile, Summary
    ("CALL AllocAry( PhaseAngles, p%grid%NPoints, p%grid%NumFreq, 3, 'Random Phases', ErrStat, ErrMsg )\n", 2),  # Spektren
    ("CALL AllocAry( V, p%grid%NumSteps, p%grid%NPoints, 3, 'V (velocity)', ErrStat, ErrMsg) !  Allocate the array that contains the velocities.\n", 3),  # Phasen
    ("CALL CalcFourierCoeffs(  p, U, PhaseAngles, S, V, ErrStat, ErrMsg )\n", 4),  # V allozieren
    ("CALL Coeffs2TimeSeries( V, p%grid%NumSteps, p%grid%NPoints, p%usr%NPoints, ErrStat, ErrMsg)\n", 5),  # Frequenzschleife (+Dump)
    ("CALL ScaleTimeSeries(p, V, ErrStat, ErrMsg)\n", 6),                           # IFFT
    ("CALL WrSum_Stats(p, V, USig, VSig, WSig, ErrStat, ErrMsg)\n", 7),            # Skalierung
    ("CALL AddMeanAndRotate(p, V, U, HWindDir, VWindDir)\n", 8),                   # Statistik + HH-Dateien
    ("IF ( p%WrFile(FileExt_BTS) .OR. p%WrFile(FileExt_WND) .OR. p%WrFile(FileExt_HAWC) ) THEN\n", 9),  # Mittelwind/Rotation (+CTS)
]
NAMEN = ["Einlesen/Gitter/Profile", "Spektren (CalcTargetPSD)", "Phasen (RANDOM_NUMBER)", "V allozieren",
         "Frequenzschleife (+Dump)", "IFFT (Coeffs2TimeSeries)", "Skalierung", "Statistik/HH-Dateien",
         "Mittelwind/Rotation", "Ausgabe .bts/.wnd/.hawc"]

ENDE_ANKER = "WRITE (p%US,\"(//,'Processing complete.  ',A,' CPU seconds used.')\")  TRIM( Num2LStr( CPUtime ) )\n"
ENDE_NEU = ("CALL SYSTEM_CLOCK( tsZeit(10) )\n"
            "CALL GET_ENVIRONMENT_VARIABLE( 'TS_ZEITEN', tsZeitAn )\n"
            "IF ( LEN_TRIM(tsZeitAn) > 0 ) THEN\n"
            "   CALL SYSTEM_CLOCK( COUNT_RATE=tsZeitRate )\n"
            "   CALL WrScr( ' TurbSim-Phasen (Wanduhr):' )\n"
            + "".join(f"   CALL WrScr( '   {n:26s}: '//TRIM(Num2LStr(REAL(tsZeit({i+1})-tsZeit({i}),8)/REAL(tsZeitRate,8)))//' s' )\n"
                      for i, n in enumerate(NAMEN))
            + "   CALL WrScr( '   gesamt                    : '//TRIM(Num2LStr(REAL(tsZeit(10)-tsZeit(0),8)/REAL(tsZeitRate,8)))//' s' )\n"
            "END IF\n"
            + ENDE_ANKER)

START_ANKER = "CALL TS_ValidateInput(p, ErrStat, ErrMsg)\n"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Phasenzeiten bereits eingebaut — weiter.")
        return
    assert DEKL_ANKER in s, "patch_turbsim_dump.py muss zuerst laufen."
    for a, _ in STEMPEL + [(ENDE_ANKER, 0), (START_ANKER, 0)]:
        assert s.count(a) == 1, f"Anker {a[:60]!r}: {s.count(a)}x"
    s = s.replace(DEKL_ANKER, DEKL_NEU, 1)
    s = s.replace(START_ANKER, "CALL SYSTEM_CLOCK( tsZeit(0) )\n" + START_ANKER, 1)
    for a, k in STEMPEL:
        s = s.replace(a, f"CALL SYSTEM_CLOCK( tsZeit({k}) )\n" + a, 1)
    s = s.replace(ENDE_ANKER, ENDE_NEU, 1)
    p.write_text(s)
    print(f"{p.name}: Phasenzeiten (TS_ZEITEN) eingebaut.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

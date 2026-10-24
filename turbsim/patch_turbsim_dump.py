#!/usr/bin/env python3
"""Schreibt die double-Fourierkoeffizienten V vor der IFFT in eine Datei (idempotent).

Zweck: Bitgleichheit auf DOUBLE-Ebene pruefen, nicht erst an der int16-.bts.
Die .bts entsteht aus V erst nach Rundung auf float32, FFT, Mittelwind und
int16-Quantisierung — ULP-Unterschiede in V sind dort meist unsichtbar. Wer
die Frequenzschleife umbaut (Gather, eigene Cholesky, GPU), braucht den
Vergleich davor.

Der Patch aendert NUR TurbSim.f90 (Hauptprogramm), nicht TSsubs.f90: die
Rechenroutinen bleiben byteweise identisch zur Referenz, der Dump ist also
ein Dump der Referenzrechnung.

Verhalten: Ist die Umgebungsvariable TS_DUMP_V gesetzt, wird direkt nach
CalcFourierCoeffs das ganze Feld V(NumSteps,NPoints,3) als Stream
(ACCESS='STREAM', native double, Fortran-Reihenfolge) in diese Datei
geschrieben, davor drei INTEGER(4): NumSteps, NPoints, 3. Ohne Variable
passiert nichts.

Verwendung:  python3 patch_turbsim_dump.py <baum>/modules/turbsim/src/TurbSim.f90
"""
import sys
from pathlib import Path

MARKE = "TS_DUMP_V"

ANKER = """CALL CalcFourierCoeffs(  p, U, PhaseAngles, S, V, ErrStat, ErrMsg )
CALL CheckError(ErrStat, ErrMsg)
"""

NEU = """CALL CalcFourierCoeffs(  p, U, PhaseAngles, S, V, ErrStat, ErrMsg )
CALL CheckError(ErrStat, ErrMsg)

   ! --- Pruefpunkt: V (double) vor der IFFT ablegen, wenn TS_DUMP_V gesetzt ---
   CALL GET_ENVIRONMENT_VARIABLE( 'TS_DUMP_V', tsDumpPfad, STATUS=tsDumpStat )
   IF ( tsDumpStat == 0 .AND. LEN_TRIM(tsDumpPfad) > 0 ) THEN
      CALL GetNewUnit( tsDumpUnit )
      OPEN( UNIT=tsDumpUnit, FILE=TRIM(tsDumpPfad), FORM='UNFORMATTED', ACCESS='STREAM', &
            STATUS='REPLACE', ACTION='WRITE', IOSTAT=tsDumpStat )
      IF ( tsDumpStat /= 0 ) CALL CheckError( ErrID_Fatal, 'TS_DUMP_V: Datei konnte nicht angelegt werden.' )
      WRITE( tsDumpUnit ) INT( SIZE(V,1), 4 ), INT( SIZE(V,2), 4 ), INT( SIZE(V,3), 4 )
      WRITE( tsDumpUnit ) V
      CLOSE( tsDumpUnit )
      CALL WrScr( ' V (double, '//TRIM(Num2LStr(SIZE(V,1)))//' x '//TRIM(Num2LStr(SIZE(V,2)))// &
                  ' x 3) nach '//TRIM(tsDumpPfad)//' geschrieben.' )
   END IF
"""

DEKL_ANKER = "REAL(ReKi), ALLOCATABLE          :: V           (:,:,:)     ! An array containing the summations of the rows of H (NumSteps,NPoints,3)."
DEKL_NEU = (
    "REAL(ReKi), ALLOCATABLE          :: V           (:,:,:)     ! An array containing the summations of the rows of H (NumSteps,NPoints,3).\n"
    "CHARACTER(1024)                  :: tsDumpPfad          ! TS_DUMP_V: Zielpfad fuer den V-Dump\n"
    "INTEGER(IntKi)                   :: tsDumpStat, tsDumpUnit"
)


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: V-Dump bereits eingebaut — weiter.")
        return
    for a in (ANKER, DEKL_ANKER):
        n = s.count(a)
        assert n == 1, f"{p.name}: Fundstring {n}x statt 1x vorhanden: {a[:60]!r}"
    s = s.replace(DEKL_ANKER, DEKL_NEU, 1)
    s = s.replace(ANKER, NEU, 1)
    p.write_text(s)
    print(f"{p.name}: V-Dump (TS_DUMP_V) eingebaut.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

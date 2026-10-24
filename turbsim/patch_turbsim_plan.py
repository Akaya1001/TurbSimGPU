#!/usr/bin/env python3
"""TS_PLAN: TurbSim meldet nur die Groessen, die RAM, VRAM und Platte bestimmen, und endet (idempotent, nur TurbSim.f90).

Fuer gpu_windfelder.py (17.10.2026): Nach CreateGrid stehen Punktzahl, Zeitschritte, Frequenzen, Ausgabegitter,
Kohaerenzmodelle und Ausgabedateien fest, alles aus TurbSims eigener Logik (kein Nachbau in Python). Mit gesetztem
TS_PLAN schreibt TurbSim eine Zeile

    TS_PLAN npoints=... npacked=... numsteps=... numfreq=... numoutsteps=... ngrid=... ny=... nz=... ntwr=... usr=...
            scmod=a,b,c cohgleich=0|1 wr=<9 Ziffern WrFile> turbmodel=... desclen=...

auf die Standardausgabe und endet regulaer, bevor S, Phasen oder V angelegt werden (Millisekunden, kein Speicher).
Ausgeloest durch einen nicht leeren Wert ausser "0" (TS_PLAN=1); ohne TS_PLAN aendert sich nichts. Steht in #ifdef TURBSIM_GPU.

Setzt patch_turbsim_zeiten.py voraus.

Verwendung:  python3 patch_turbsim_plan.py <baum>/modules/turbsim/src/TurbSim.f90
"""
import sys
from pathlib import Path

MARKE = "TS_PLAN"

DEKL_ANKER = "CHARACTER(16)                    :: tsZeitAn\n"
DEKL_NEU = (DEKL_ANKER +
            "#ifdef TURBSIM_GPU\n"
            "CHARACTER(16)                    :: tsPlanAn            ! TS_PLAN: nur Groessen melden\n"
            "CHARACTER(NumFileFmt)            :: tsPlanWr\n"
            "INTEGER(IntKi)                   :: tsPlanStat, tsPlanLen, tsPlanTwr, tsPlanI\n"
            "LOGICAL                          :: tsPlanGleich\n"
            "#endif\n")

ANKER = ("CALL CreateGrid( p%grid, p%usr, p%UHub, p%WrFile(FileExt_TWR), ErrStat, ErrMsg )\n"
         "CALL CheckError(ErrStat, ErrMsg)\n")


def _z(ausdruck: str) -> str:
    return f"TRIM(Num2LStr({ausdruck}))"


NEU = (ANKER +
       "#ifdef TURBSIM_GPU\n"
       "   ! TS_PLAN (turbsim/gpu_windfelder.py): Groessen fuer die Planung melden, dann ohne Rechnung enden.\n"
       "   ! gesetzt und nicht leer, nicht '0' (auch ein ueberlanger Wert, STATUS=-1, zaehlt als gesetzt)\n"
       "CALL GET_ENVIRONMENT_VARIABLE( 'TS_PLAN', tsPlanAn, LENGTH=tsPlanLen, STATUS=tsPlanStat )\n"
       "IF ( (tsPlanStat == 0 .OR. tsPlanStat == -1) .AND. tsPlanLen > 0 .AND. tsPlanAn(1:1) /= '0' ) THEN\n"
       "   tsPlanTwr = 0\n"
       "   IF ( p%WrFile(FileExt_TWR) .AND. ALLOCATED(p%grid%TwrPtIndx) ) tsPlanTwr = SIZE(p%grid%TwrPtIndx)\n"
       "   DO tsPlanI = 1, NumFileFmt\n"
       "      tsPlanWr(tsPlanI:tsPlanI) = MERGE('1', '0', p%WrFile(tsPlanI))\n"
       "   END DO\n"
       "   tsPlanGleich = ALL( p%met%SCMod == CohMod_IEC ) .AND. &\n"
       "                  p%met%InCDec(1) == p%met%InCDec(2) .AND. p%met%InCDec(2) == p%met%InCDec(3) .AND. &\n"
       "                  p%met%InCohB(1) == p%met%InCohB(2) .AND. p%met%InCohB(2) == p%met%InCohB(3)\n"
       "   WRITE(*,'(A)') 'TS_PLAN npoints='//" + _z("p%grid%NPoints") + "//' npacked='//" + _z("p%grid%NPacked") + "// &\n"
       "        ' numsteps='//" + _z("p%grid%NumSteps") + "//' numfreq='//" + _z("p%grid%NumFreq") + "// &\n"
       "        ' numoutsteps='//" + _z("p%grid%NumOutSteps") + "//' ngrid='//" + _z("SIZE(p%grid%GridPtIndx)") + "// &\n"
       "        ' ny='//" + _z("p%grid%NumGrid_Y") + "//' nz='//" + _z("p%grid%NumGrid_Z") + "//' ntwr='//" + _z("tsPlanTwr") + "// &\n"
       "        ' usr='//" + _z("p%usr%NPoints") + "// &\n"
       "        ' scmod='//" + _z("p%met%SCMod(1)") + "//','//" + _z("p%met%SCMod(2)") + "//','//" + _z("p%met%SCMod(3)") + "// &\n"
       "        ' cohgleich='//MERGE('1', '0', tsPlanGleich)//' wr='//tsPlanWr// &\n"
       "        ' turbmodel='//" + _z("p%met%TurbModel_ID") + "//' desclen='//" + _z("LEN_TRIM(p%DescStr)") + "\n"
       "   CALL NormStop()\n"
       "END IF\n"
       "#endif\n")


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: TS_PLAN bereits eingebaut — weiter.")
        return
    assert DEKL_ANKER in s, "patch_turbsim_zeiten.py muss zuerst laufen."
    assert s.count(DEKL_ANKER) == 1 and s.count(ANKER) == 1
    s = s.replace(DEKL_ANKER, DEKL_NEU, 1).replace(ANKER, NEU, 1)
    p.write_text(s)
    print(f"{p.name}: TS_PLAN eingebaut (#ifdef TURBSIM_GPU).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

#!/usr/bin/env python3
"""Nur-GPU-Betrieb (TS_GPU=nur, gpu_windfelder.py, 17.10.2026): CPU-Pfad der Kohaerenzschleife sperren (idempotent).

Mit TS_GPU=nur meldet tsg_pflicht() (turbsim/gpu/tsg_host.hip) 1. Dann bricht TurbSim mit Fehler ab, statt eine
Frequenzschleife auf der CPU zu rechnen:
  * CalcFourierCoeffs: Kohaerenzmodell GENERAL oder API (rechnet nur die CPU; die GPU kann nur IEC);
  * CalcFourierCoeffs_IEC: GPU-Lauf abgelehnt (keine Karte, VRAM, S-Klassen, HIP-Fehler) oder mehr als ein Nutzerpunkt.
Die Nachrechnung einer einzelnen Fehlfrequenz (nicht positiv definit) bleibt erlaubt: sie ist eine Zerlegung und erzeugt
nur den Fehlertext der Referenz. Ohne TS_GPU=nur aendert sich nichts. Alles steht in #ifdef TURBSIM_GPU.

Setzt patch_turbsim_gpu.py voraus.

Verwendung:  python3 patch_turbsim_gpunur.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsg_pflicht"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Nur-GPU-Sperre bereits eingebaut — weiter.")
        return
    assert "tsg_iec_lauf" in s, "patch_turbsim_gpu.py muss zuerst laufen."

    # --- 1. Schnittstelle --------------------------------------------------------
    alt = "      END SUBROUTINE tsg_iec_lauf\n"
    assert s.count(alt) == 1
    s = s.replace(alt, alt +
        "      FUNCTION tsg_pflicht() BIND(C, NAME=\"tsg_pflicht\")\n"
        "         INTEGER(4) :: tsg_pflicht\n"
        "      END FUNCTION tsg_pflicht\n", 1)

    # --- 2. GENERAL/API sperren (Verteiler) ---------------------------------------
    alt = "   CALL CalcFourierCoeffs_IEC(     p, U, PhaseAngles, S, V, TRH, ErrStat2, ErrMsg2 );"
    neu = ("#ifdef TURBSIM_GPU\n"
           "   IF (tsg_pflicht() == 1 .AND. ANY(p%met%SCMod == CohMod_GENERAL .OR. p%met%SCMod == CohMod_API)) THEN\n"
           "      CALL SetErrStat( ErrID_Fatal, 'TS_GPU=nur: Kohaerenzmodell GENERAL/API rechnet nur die CPU, '// &\n"
           "           'und die ist gesperrt (die GPU kann nur IEC).', ErrStat, ErrMsg, 'CalcFourierCoeffs' )\n"
           "      CALL Cleanup()\n"
           "      RETURN\n"
           "   END IF\n"
           "#endif\n" + alt)
    assert s.count(alt) == 1                   # nur im Verteiler (ersetze_in_routine traefe auch ..._IEC usw.)
    s = s.replace(alt, neu, 1)

    # --- 3. Abgelehnten GPU-Lauf nicht auf der CPU nachholen -------------------------
    alt = ("                            V, SIZE(V,1), SIZE(V,2), IVec, tsGpuNComp, tsGpuFehl, tsGpuOk )\n"
           "      END IF\n")
    neu = (alt +
           "      IF (tsGpuOk /= 1 .AND. tsg_pflicht() == 1) THEN\n"
           "         IF (p%usr%NPoints > 1) THEN\n"
           "            CALL SetErrStat( ErrID_Fatal, 'TS_GPU=nur: mehr als ein Nutzerpunkt - das rechnet nur die CPU, '// &\n"
           "                 'und die ist gesperrt.', ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC' )\n"
           "         ELSE\n"
           "            CALL SetErrStat( ErrID_Fatal, 'TS_GPU=nur: die GPU hat die Frequenzschleife abgelehnt (Grund in der '// &\n"
           "                 'Meldung \"TurbSim-GPU:\"), der CPU-Pfad ist gesperrt.', ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC' )\n"
           "         END IF\n"
           "         CALL Cleanup()\n"
           "         RETURN\n"
           "      END IF\n")
    s = ersetze_in_routine(s, "CalcFourierCoeffs_IEC", alt, neu)

    p.write_text(s)
    print(f"{p.name}: Nur-GPU-Sperre eingebaut (TS_GPU=nur, #ifdef TURBSIM_GPU).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

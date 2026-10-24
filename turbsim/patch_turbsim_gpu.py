#!/usr/bin/env python3
"""Frequenzschleife der IEC-Kohaerenz auf der GPU (Stufe G3, Plan 14.10.2026), bitgleich (idempotent).

Vor der OpenMP-Schleife in CalcFourierCoeffs_IEC ruft TurbSim den GPU-Motor `tsg_iec_lauf` (turbsim/gpu/
tsg_host.hip): Gather, Cholesky mit zertifiziertem Auslassen, S-Skalierung und H2Coeffs fuer alle Frequenzen dieser
Komponente (bei identischen u/v/w-Parametern fuer alle drei). Danach:
  ok = 1, keine Fehlfrequenz -> CPU-Schleife entfaellt fuer diese Komponente(n);
  ok = 1, Fehlfrequenz f     -> CPU-Schleife nur fuer f: sie scheitert wie die Referenz und erzeugt deren Fehlertext
                                (rechnet sie wider Erwarten ohne Fehler, bricht TurbSim mit einer eigenen Meldung ab);
  ok = 0 (keine GPU, zu wenig VRAM, TS_GPU=0, Nutzerpunkte > 1) -> unveraenderter CPU-Pfad, V vorher neu genullt.
Alles steht in #ifdef TURBSIM_GPU; ohne das Makro ist der Quelltext unveraendert wirksam.

Setzt die Stufe-1-Kette voraus (laeuft nach patch_turbsim_fehlerpfad.py).

Verwendung:  python3 patch_turbsim_gpu.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsg_iec_lauf"
R = "CalcFourierCoeffs_IEC"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: GPU-Pfad bereits eingebaut — weiter.")
        return
    assert "tsAbbIFreq" in s and "ts_h2c" in s, "Stufe-1-Kette inkl. patch_turbsim_fehlerpfad.py muss zuerst laufen."

    # --- 1. Schnittstelle (im bestehenden INTERFACE-Block) ------------------
    alt = "   END INTERFACE"
    assert s.count(alt) == 1
    s = s.replace(alt,
        "#ifdef TURBSIM_GPU\n"
        "      SUBROUTINE tsg_iec_lauf( n, nfreq, cls, k, distk, distuk, uhub, freq, incdec, incohb, s, nfs, nps, &\n"
        "                               phase, ldp, nfp, v, ldv, npv, ivec, ncomp, fehl_ifreq, ok ) BIND(C, NAME=\"tsg_iec_lauf\")\n"
        "         INTEGER(4), INTENT(IN)    :: n, nfreq, cls(*), k, nfs, nps, ldp, nfp, ldv, npv, ivec, ncomp\n"
        "         REAL(8),    INTENT(IN)    :: distk(*), distuk(*), uhub, freq(*), incdec, incohb, s(*), phase(*)\n"
        "         REAL(8),    INTENT(INOUT) :: v(*)\n"
        "         INTEGER(4), INTENT(OUT)   :: fehl_ifreq, ok\n"
        "      END SUBROUTINE tsg_iec_lauf\n"
        "#endif\n"
        "   END INTERFACE", 1)

    # --- 2. Lokale Groessen -------------------------------------------------
    s = ersetze_in_routine(s, R,
        "INTEGER                       :: tsChunk       ! OpenMP-Chunk, an NumFreq angepasst\n",
        "INTEGER                       :: tsChunk       ! OpenMP-Chunk, an NumFreq angepasst\n"
        "INTEGER(4)                    :: tsGpuOk, tsGpuFehl, tsGpuNComp\n"
        "INTEGER                       :: tsIF0, tsIF1   ! Frequenzbereich der CPU-Schleife\n")

    # --- 3. GPU-Aufruf vor der Schleife ---------------------------------------
    alt = "      tsChunk     = MAX(1, MIN(64, p%grid%NumFreq/48))\n"
    neu = ("      tsChunk     = MAX(1, MIN(64, p%grid%NumFreq/48))\n"
           "      tsIF0 = 1\n"
           "      tsIF1 = p%grid%NumFreq\n"
           "#ifdef TURBSIM_GPU\n"
           "      ! Ganze Frequenzschleife auf der GPU (turbsim/gpu/tsg_host.hip); Nutzerpunkte > 1 nur auf der CPU.\n"
           "      tsGpuOk = 0\n"
           "      tsGpuFehl = 0\n"
           "      IF (p%usr%NPoints <= 1) THEN\n"
           "         tsGpuNComp = MERGE(3, 1, tsAlleGleich)\n"
           "         CALL tsg_iec_lauf( tsN, p%grid%NumFreq, tsCls, tsK, tsDistK, tsDistUK, p%UHub, p%grid%Freq, &\n"
           "                            p%met%InCDec(IVec), p%met%InCohB(IVec), S, SIZE(S,1), SIZE(S,2), &\n"
           "                            PhaseAngles, SIZE(PhaseAngles,1), SIZE(PhaseAngles,2), &\n"
           "                            V, SIZE(V,1), SIZE(V,2), IVec, tsGpuNComp, tsGpuFehl, tsGpuOk )\n"
           "      END IF\n"
           "      IF (tsGpuOk == 1 .AND. tsGpuFehl == 0) THEN\n"
           "         CALL WrScr( '    Frequenzschleife auf der GPU gerechnet' )\n"
           "         CYCLE\n"
           "      ELSE IF (tsGpuOk == 1) THEN\n"
           "         ! Diese Frequenz auf dem CPU-Pfad nachrechnen: Fehlertext byteweise wie die Referenz\n"
           "         tsIF0 = tsGpuFehl\n"
           "         tsIF1 = tsGpuFehl\n"
           "      ELSE\n"
           "         V(:,:,IVec) = 0.0_ReKi                ! die GPU kann V teilweise beschrieben haben\n"
           "         IF (tsAlleGleich) V(:,:,2:3) = 0.0_ReKi\n"
           "      END IF\n"
           "#endif\n")
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 4. Schleifengrenzen ----------------------------------------------------
    s = ersetze_in_routine(s, R,
        "      DO IFREQ = 1,p%grid%NumFreq\n",
        "      DO IFREQ = tsIF0,tsIF1\n")

    # --- 5. Widerspruch GPU/CPU abfangen (nach der Auswertung der Fehler) ---------
    alt = ("      IF (tsFehlIFreq /= HUGE(1_IntKi)) THEN\n"
           "         CALL SetErrStat(tsFehlStat, tsFehlMsg, ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC')\n"
           "      END IF\n")
    neu = (alt +
           "#ifdef TURBSIM_GPU\n"
           "      IF (tsGpuOk == 1 .AND. tsGpuFehl > 0 .AND. tsFehlIFreq == HUGE(1_IntKi)) THEN\n"
           "         CALL SetErrStat( ErrID_Fatal, 'GPU meldete eine nicht positiv definite Matrix bei Frequenz '// &\n"
           "              TRIM(Int2LStr(tsGpuFehl))//', die CPU nicht - Widerspruch, Lauf abgebrochen.', ErrStat, ErrMsg, &\n"
           "              'CalcFourierCoeffs_IEC' )\n"
           "      END IF\n"
           "#endif\n")
    s = ersetze_in_routine(s, R, alt, neu)

    p.write_text(s)
    print(f"{p.name}: GPU-Pfad eingebaut (#ifdef TURBSIM_GPU).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

#!/usr/bin/env python3
"""Kostenmessung fuer TurbSims Fourier-Koeffizienten (idempotent).

Misst die drei Bloecke der Frequenzschleife getrennt:
  Kohaerenzaufbau (exp/sqrt ueber NPacked)  ->  Coh2H (Cholesky)  ->  H2Coeffs
sowie den Identitaetspfad (v-/w-Komponente bei den IEC-Turbulenzmodellen).

Anlass: ein frueherer Cholesky-Patch brachte auf der Einzel-
turbinen-Box nur 1,06x, auf der Farm-High-Box aber 2,63x. Welcher Block
die restliche Zeit haelt, war nicht gemessen. Gemessen am 29.08.2026:
Kohaerenzaufbau ~60 %, Cholesky ~31 %, H2Coeffs ~4 %, Identitaet ~5 %; die
Schleife ist 98,6 % der Wanduhr.

Die Zeitstempel liegen bewusst LOKAL in den Routinen (nicht als SAVE auf
Modulebene), damit patch_turbsim_omp.py die Schleife parallelisieren kann;
die Summierer werden mit !$OMP ATOMIC bedient (ohne -fopenmp ein Kommentar).

Verwendung:  python3 patch_turbsim_instr.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsMess_Coh"

KOPF_ALT = """   USE NWTC_LAPACK


   IMPLICIT NONE



CONTAINS"""

KOPF_NEU = """   USE NWTC_LAPACK


   IMPLICIT NONE

   ! --- Kostenmessung der Frequenzschleife -----------------------------
   ! Nur die Summierer stehen hier; die Zeitstempel sind in den Routinen
   ! lokal, damit die Schleife parallelisierbar bleibt.
   INTEGER(8), SAVE :: tsMess_Coh  = 0_8   !< Kohaerenzaufbau (exp/sqrt)
   INTEGER(8), SAVE :: tsMess_Chol = 0_8   !< Coh2H inkl. Cholesky
   INTEGER(8), SAVE :: tsMess_H2C  = 0_8   !< H2Coeffs
   INTEGER(8), SAVE :: tsMess_Eye  = 0_8   !< Identitaetspfad
   INTEGER(8), SAVE :: tsMess_Rate = 0_8



CONTAINS

!=======================================================================
!> Gibt die Kostenmessung der Frequenzschleife aus. Einmal je Lauf.
SUBROUTINE TS_Messung_Ausgeben( Etikett, NPoints, NumFreq )
   CHARACTER(*),   INTENT(IN) :: Etikett
   INTEGER(IntKi), INTENT(IN) :: NPoints, NumFreq
   REAL(8) :: r, tc, tf, th, te
   IF (tsMess_Rate <= 0_8) CALL SYSTEM_CLOCK( COUNT_RATE=tsMess_Rate )
   r  = REAL(tsMess_Rate,8)
   tc = REAL(tsMess_Coh, 8)/r
   tf = REAL(tsMess_Chol,8)/r
   th = REAL(tsMess_H2C, 8)/r
   te = REAL(tsMess_Eye, 8)/r
   WRITE(*,'(A)')         ' TurbSim-Kostenmessung ('//TRIM(Etikett)//'):'
   WRITE(*,'(A,I8,A,I8)') '   NPoints = ', NPoints, '   NumFreq = ', NumFreq
   WRITE(*,'(A,F12.2,A)') '   Kohaerenzaufbau : ', tc, ' s'
   WRITE(*,'(A,F12.2,A)') '   Coh2H (Cholesky): ', tf, ' s'
   WRITE(*,'(A,F12.2,A)') '   H2Coeffs        : ', th, ' s'
   WRITE(*,'(A,F12.2,A)') '   Identitaetspfad : ', te, ' s'
   WRITE(*,'(A,F12.2,A)') '   Summe           : ', tc+tf+th+te, ' s'
END SUBROUTINE TS_Messung_Ausgeben"""


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: bereits instrumentiert — weiter.")
        return

    assert s.count(KOPF_ALT) == 1, "TSsubs.f90: Modulkopf hat unerwartete Struktur"
    s = s.replace(KOPF_ALT, KOPF_NEU, 1)

    # ---------------- IEC-Routine ----------------------------------------
    R = "CalcFourierCoeffs_IEC"

    s = ersetze_in_routine(s, R,
        "INTEGER(IntKi)                :: ErrStat2\n"
        "CHARACTER(MaxMsgLen)          :: ErrMsg2",
        "INTEGER(IntKi)                :: ErrStat2\n"
        "CHARACTER(MaxMsgLen)          :: ErrMsg2\n"
        "INTEGER(8)                    :: tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3")

    s = ersetze_in_routine(s, R,
        "      DO IFREQ = 1,p%grid%NumFreq\n"
        "         ! -----------------------------------------------\n"
        "         ! Create the coherence matrix for this frequency",
        "      DO IFREQ = 1,p%grid%NumFreq\n"
        "         CALL SYSTEM_CLOCK( tsMess_c0, tsMess_Rate )\n"
        "         ! -----------------------------------------------\n"
        "         ! Create the coherence matrix for this frequency")

    s = ersetze_in_routine(s, R,
        "         CALL Coh2H(    p, IVec, IFreq, TRH, S, ErrStat2, ErrMsg2 )",
        "         CALL SYSTEM_CLOCK( tsMess_c1 )\n"
        "         CALL Coh2H(    p, IVec, IFreq, TRH, S, ErrStat2, ErrMsg2 )")

    s = ersetze_in_routine(s, R,
        "         CALL H2Coeffs( IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )\n"
        "      END DO !IFreq",
        "         CALL SYSTEM_CLOCK( tsMess_c2 )\n"
        "         CALL H2Coeffs( IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )\n"
        "         CALL SYSTEM_CLOCK( tsMess_c3 )\n"
        "         !$OMP ATOMIC\n"
        "         tsMess_Coh  = tsMess_Coh  + (tsMess_c1 - tsMess_c0)\n"
        "         !$OMP ATOMIC\n"
        "         tsMess_Chol = tsMess_Chol + (tsMess_c2 - tsMess_c1)\n"
        "         !$OMP ATOMIC\n"
        "         tsMess_H2C  = tsMess_H2C  + (tsMess_c3 - tsMess_c2)\n"
        "      END DO !IFreq")

    # ---------------- Identitaetsroutine ----------------------------------
    # Die Ausgabe steht hier und nicht in der IEC-Routine: _NONE laeuft
    # spaeter, der IEC-Ausdruck saehe tsMess_Eye sonst immer als 0.
    N = "CalcFourierCoeffs_NONE"

    s = ersetze_in_routine(s, N,
        "INTEGER                       :: IVec  ! wind component, 1=u, 2=v, 3=w",
        "INTEGER                       :: IVec  ! wind component, 1=u, 2=v, 3=w\n"
        "INTEGER(8)                    :: tsMess_e0, tsMess_e1")

    s = ersetze_in_routine(s, N,
        "      DO IFREQ = 1,p%grid%NumFreq\n"
        "         CALL EyeCoh2H(  IVec, IFreq, TRH, S,              p%grid%NPoints )   \n"
        "         CALL H2Coeffs(  IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )\n"
        "      ENDDO !IFreq",
        "      CALL SYSTEM_CLOCK( tsMess_e0, tsMess_Rate )\n"
        "      DO IFREQ = 1,p%grid%NumFreq\n"
        "         CALL EyeCoh2H(  IVec, IFreq, TRH, S,              p%grid%NPoints )   \n"
        "         CALL H2Coeffs(  IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )\n"
        "      ENDDO !IFreq\n"
        "      CALL SYSTEM_CLOCK( tsMess_e1 )\n"
        "      tsMess_Eye = tsMess_Eye + (tsMess_e1 - tsMess_e0)")

    s = ersetze_in_routine(s, N,
        "   END DO ! IVec\n\n   RETURN",
        "   END DO ! IVec\n\n"
        "   CALL TS_Messung_Ausgeben( 'gesamt', p%grid%NPoints, p%grid%NumFreq )\n"
        "   RETURN")

    p.write_text(s)
    print(f"{p.name}: instrumentiert.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

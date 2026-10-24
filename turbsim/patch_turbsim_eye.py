#!/usr/bin/env python3
"""Identitaetspfad (SCMod = NONE, bei IECKAI die v- und w-Komponente) in
geschlossener Form O(N) statt O(N^2) — bitgleich, OpenMP ueber die Frequenzen
(idempotent).

Beweis der Bitgleichheit: EyeCoh2H setzt die Diagonale SQRT(ABS(S)) und alle
Nebendiagonalelemente 0.0; H2Coeffs summiert je Zeile I in aufsteigendem J:
   V = ((((+0) + 0*CPh_1) + 0*CPh_2) + ...) + SQRT(ABS(S_I))*CPh_I
Jedes Produkt 0*CPh_J ist +-0; IEEE-RN: (+0) + (+-0) = +0. Die Teilsumme vor dem
Diagonalbeitrag ist also exakt +0, und der letzte Schritt lautet
   V = (+0) + SQRT(ABS(S_I))*CPh_I
— genau das rechnet die geschlossene Form (einschliesslich Nullvorzeichen, weil
V weiterhin mit +0 vorbelegt wird und die Addition ausgeschrieben bleibt).
COS/SIN rechnet der C-Kern ts_eye skalar mit glibc sincos wie die Referenz; eine
Fortran-Schleife ueber COS/SIN wuerde gfortran mit libmvec vektorisieren (andere Bits).

Setzt patch_turbsim_instr.py und patch_turbsim_gather.py voraus.

Verwendung:  python3 patch_turbsim_eye.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsEyeI"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Identitaetspfad bereits O(N) — weiter.")
        return
    assert "tsMess_e0" in s, "patch_turbsim_instr.py muss zuerst laufen."
    assert "   END INTERFACE" in s, "patch_turbsim_gather.py muss zuerst laufen (INTERFACE-Block)."
    s = s.replace("   END INTERFACE",
        "      SUBROUTINE ts_eye( n, s, nfs, nps, phase, ldp, nfp, ifreq, v, ldv, npv, if1, ivec ) BIND(C, NAME=\"ts_eye\")\n"
        "         INTEGER(4), INTENT(IN)    :: n, nfs, nps, ldp, nfp, ifreq, ldv, npv, if1, ivec\n"
        "         REAL(8),    INTENT(IN)    :: s(*), phase(*)\n"
        "         REAL(8),    INTENT(INOUT) :: v(*)\n"
        "      END SUBROUTINE ts_eye\n"
        "   END INTERFACE", 1)
    N = "CalcFourierCoeffs_NONE"

    s = ersetze_in_routine(s, N,
        "INTEGER(8)                    :: tsMess_e0, tsMess_e1",
        "INTEGER(8)                    :: tsMess_e0, tsMess_e1\n"
        "INTEGER(4)                    :: tsEyeI, tsIF1, tsEyeNfs, tsEyeNps, tsEyeLdp, tsEyeNfp, tsEyeLdv, tsEyeNpv, tsEyeChunk")

    alt = ("      CALL SYSTEM_CLOCK( tsMess_e0, tsMess_Rate )\n"
           "      DO IFREQ = 1,p%grid%NumFreq\n"
           "         CALL EyeCoh2H(  IVec, IFreq, TRH, S,              p%grid%NPoints )   \n"
           "         CALL H2Coeffs(  IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )\n"
           "      ENDDO !IFreq\n"
           "      CALL SYSTEM_CLOCK( tsMess_e1 )")
    neu = ("      CALL SYSTEM_CLOCK( tsMess_e0, tsMess_Rate )\n"
           "      ! Geschlossene Form in C (ts_eye): H ist diagonal, die Zeilensumme von H2Coeffs ist\n"
           "      ! (+0) + SQRT(ABS(S_I))*CPh_I — bitgleich zum O(N^2)-Weg; sincos skalar aus glibc\n"
           "      ! (eine Fortran-COS/SIN-Schleife wuerde gfortran mit libmvec vektorisieren).\n"
           "      tsEyeNfs = SIZE(S,1); tsEyeNps = SIZE(S,2); tsEyeLdp = SIZE(PhaseAngles,1); tsEyeNfp = SIZE(PhaseAngles,2)\n"
           "      tsEyeLdv = SIZE(V,1); tsEyeNpv = SIZE(V,2)\n"
           "      tsEyeChunk = MAX(1, MIN(64, p%grid%NumFreq/48))\n"
           "      !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(IFreq, tsIF1) SCHEDULE(DYNAMIC,tsEyeChunk)\n"
           "      DO IFREQ = 1,p%grid%NumFreq\n"
           "         tsIF1 = IFreq*2 - 1\n"
           "         CALL ts_eye( p%grid%NPoints, S, tsEyeNfs, tsEyeNps, PhaseAngles, tsEyeLdp, tsEyeNfp, IFreq, &\n"
           "                      V, tsEyeLdv, tsEyeNpv, tsIF1, IVec )\n"
           "      ENDDO !IFreq\n"
           "      !$OMP END PARALLEL DO\n"
           "      CALL SYSTEM_CLOCK( tsMess_e1 )")
    s = ersetze_in_routine(s, N, alt, neu)

    p.write_text(s)
    print(f"{p.name}: Identitaetspfad in geschlossener Form O(N).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

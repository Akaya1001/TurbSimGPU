#!/usr/bin/env python3
"""Ersetzt LAPACK_pptrf in Coh2H durch die bitgleiche Cholesky ts_chol_spec (idempotent).

Spezifikation (verifiziert gegen OpenBLAS dpptrf_, 0 Abweichungen inkl.
Nullvorzeichen, n bis 1225, alle ISA-Fassungen): je Element (i,c) die
FMA-Kette ueber k aufsteigend mit Akkumulator aus A, Spalten mit L(c,k)==0
ausgelassen, r = 1/L(c,c), L(i,c) = s*r. Der C-Kern (turbsim/kern/)
rechnet das geblockt (Panel 256, rekursiv bis 16) mit AVX-512/AVX2 und
Laufzeit-Dispatch; die Rechenreihenfolge je Element ist dieselbe wie bei
dpptrf, deshalb bitgleich.

Zeilenprofil (frow aus dem Gather): Zeilen, deren erster Eintrag != 0 hinter
dem Panel liegt, haben dort L = 0 und werden im Nachlauf-Update ausgelassen
(wertgleich; nur Nullvorzeichen koennen abweichen — V sieht sie nie). Das
Profil wird nur genutzt, wenn die Cholesky ueber die volle Matrix laeuft
(usr%NPoints <= 1).

Die Fehlermeldung wird byteweise wie NWTC_LAPACK.f90 (LAPACK_DPPTRF) gebaut,
damit Coh2H und der Aufrufer unveraendert reagieren.

Setzt patch_turbsim_gather.py voraus (INTERFACE-Block, tsFrow).

Verwendung:  python3 patch_turbsim_chol.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "ts_chol_spec"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: ts_chol_spec bereits eingebaut — weiter.")
        return
    assert "ts_coh_fill" in s, "patch_turbsim_gather.py muss zuerst laufen."

    # --- 0. Schnittstelle ---------------------------------------------------
    alt = "      END SUBROUTINE ts_coh_fill\n   END INTERFACE"
    assert s.count(alt) == 1
    s = s.replace(alt,
        "      END SUBROUTINE ts_coh_fill\n"
        "      SUBROUTINE ts_chol_spec( n, ap, frow, use_frow, info ) BIND(C, NAME=\"ts_chol_spec\")\n"
        "         INTEGER(4), INTENT(IN)    :: n, frow(*), use_frow\n"
        "         REAL(8),    INTENT(INOUT) :: ap(*)\n"
        "         INTEGER(4), INTENT(OUT)   :: info\n"
        "      END SUBROUTINE ts_chol_spec\n"
        "   END INTERFACE", 1)

    # --- 1. Coh2H: optionales Zeilenprofil, Aufruf, Fehlertext ------------
    C = "Coh2H"
    s = ersetze_in_routine(s, C,
        "SUBROUTINE Coh2H( p, IVec, IFreq, TRH, S, ErrStat, ErrMsg )",
        "SUBROUTINE Coh2H( p, IVec, IFreq, TRH, S, ErrStat, ErrMsg, frow )")
    s = ersetze_in_routine(s, C,
        "INTEGER(IntKi),              INTENT(OUT)    :: ErrStat\n"
        "CHARACTER(*),                INTENT(OUT)    :: ErrMsg\n",
        "INTEGER(IntKi),              INTENT(OUT)    :: ErrStat\n"
        "CHARACTER(*),                INTENT(OUT)    :: ErrMsg\n"
        "INTEGER(4), OPTIONAL,        INTENT(IN)     :: frow(:)                                  ! erste Spalte != 0 je Zeile (Gather), nur bei voller Matrix\n"
        "INTEGER(4)                                  :: tsInfo, tsFrow0(1)\n")
    s = ersetze_in_routine(s, C,
        "   CALL LAPACK_pptrf( 'L', NPts, TRH(Indx:), ErrStat, ErrMsg )  ! 'L'ower triangular 'TRH' matrix (packed form), of order 'NPoints'; returns Stat",
        "   ! Bitgleiche Cholesky nach der dpptrf-Spezifikation (turbsim/kern/ts_kern.c)\n"
        "   IF ( PRESENT(frow) .AND. NPts == p%grid%NPoints ) THEN\n"
        "      CALL ts_chol_spec( NPts, TRH(Indx:), frow, 1_4, tsInfo )\n"
        "   ELSE\n"
        "      tsFrow0 = 0\n"
        "      CALL ts_chol_spec( NPts, TRH(Indx:), tsFrow0, 0_4, tsInfo )\n"
        "   END IF\n"
        "   ! Fehlertext byteweise wie NWTC_LAPACK.f90 (LAPACK_DPPTRF)\n"
        "   IF (tsInfo /= 0) THEN\n"
        "      ErrStat = ErrID_FATAL\n"
        "      WRITE( ErrMsg, * ) tsInfo\n"
        "      IF (tsInfo < 0) THEN\n"
        "         ErrMsg  = \"LAPACK_DPPTRF: illegal value in argument \"//TRIM(ErrMsg)//\".\"\n"
        "      ELSE\n"
        "         ErrMsg = 'LAPACK_DPPTRF: Leading minor order '//TRIM(ErrMsg)//' of A is not positive definite, so Cholesky factorization could not be completed.'\n"
        "      END IF\n"
        "   ELSE\n"
        "      ErrStat = ErrID_None\n"
        "      ErrMsg  = \"\"\n"
        "   END IF")

    # --- 2. IEC-Schleife: Zeilenprofil mitgeben ----------------------------
    s = ersetze_in_routine(s, "CalcFourierCoeffs_IEC",
        "         CALL Coh2H(    p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3 )",
        "         CALL Coh2H(    p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3, tsFrow )")

    p.write_text(s)
    print(f"{p.name}: LAPACK_pptrf durch ts_chol_spec ersetzt.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

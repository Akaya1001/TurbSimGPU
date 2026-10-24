#!/usr/bin/env python3
"""Eine Cholesky je Frequenz fuer u, v und w, wenn alle drei Komponenten die
IEC-Kohaerenz mit identischen Parametern fahren (Windpark-Boxen: SCMod1-3 = IEC,
InCDec/InCohB gleich) — bitgleich (idempotent).

Begruendung: Der Kohaerenzaufbau haengt nur von Freq, Dist, InCDec(IVec) und
InCohB(IVec) ab; die Spektren S kommen erst NACH der Zerlegung (Skalierung in
Coh2H) hinein. Sind die Parameter fuer u, v, w gleich, ist die Matrix und damit
der Faktor L fuer alle drei bitweise derselbe. Die Referenz rechnet ihn dreimal
aus; hier wird er einmal gerechnet, kopiert und dreimal skaliert. Jede V-Scheibe
V(:,:,IVec) entsteht aus derselben Elementfolge wie zuvor; nur die Programm-
reihenfolge aendert sich (V-Scheiben sind disjunkt).

Setzt patch_turbsim_chol.py voraus.

Verwendung:  python3 patch_turbsim_farm3.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsAlleGleich"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: gemeinsame Cholesky fuer u,v,w bereits eingebaut — weiter.")
        return
    assert "ts_chol_spec( NPts, TRH(Indx:), frow, 1_4, tsInfo )" in s, "patch_turbsim_chol.py muss zuerst laufen."

    # --- 1. Coh2H: L-Kopie herausgeben bzw. nur skalieren --------------------
    C = "Coh2H"
    s = ersetze_in_routine(s, C,
        "SUBROUTINE Coh2H( p, IVec, IFreq, TRH, S, ErrStat, ErrMsg, frow )",
        "SUBROUTINE Coh2H( p, IVec, IFreq, TRH, S, ErrStat, ErrMsg, frow, LKopie, NurSkalieren )")
    s = ersetze_in_routine(s, C,
        "INTEGER(4)                                  :: tsInfo, tsFrow0(1)\n",
        "INTEGER(4)                                  :: tsInfo, tsFrow0(1)\n"
        "REAL(ReKi), OPTIONAL,        INTENT(OUT)    :: LKopie(:)                                ! Faktor L vor der S-Skalierung (fuer weitere Komponenten)\n"
        "LOGICAL,    OPTIONAL,        INTENT(IN)     :: NurSkalieren                             ! TRH enthaelt bereits L: Zerlegung ueberspringen\n"
        "LOGICAL                                     :: tsNur\n")
    alt = ("   ! Bitgleiche Cholesky nach der dpptrf-Spezifikation (turbsim/kern/ts_kern.c)\n"
           "   IF ( PRESENT(frow) .AND. NPts == p%grid%NPoints ) THEN\n")
    neu = ("   tsNur = .FALSE.\n"
           "   IF ( PRESENT(NurSkalieren) ) tsNur = NurSkalieren\n"
           "   tsInfo = 0\n"
           "   IF ( tsNur ) THEN\n"
           "      ErrStat = ErrID_None\n"
           "      ErrMsg  = \"\"\n"
           "   ! Bitgleiche Cholesky nach der dpptrf-Spezifikation (turbsim/kern/ts_kern.c)\n"
           "   ELSE IF ( PRESENT(frow) .AND. NPts == p%grid%NPoints ) THEN\n")
    s = ersetze_in_routine(s, C, alt, neu)
    alt = ("   ! Fehlertext byteweise wie NWTC_LAPACK.f90 (LAPACK_DPPTRF)\n"
           "   IF (tsInfo /= 0) THEN\n")
    neu = ("   ! Fehlertext byteweise wie NWTC_LAPACK.f90 (LAPACK_DPPTRF)\n"
           "   IF (tsNur) THEN\n"
           "      CONTINUE\n"
           "   ELSE IF (tsInfo /= 0) THEN\n")
    s = ersetze_in_routine(s, C, alt, neu)
    alt = ("   ELSE\n"
           "      ErrStat = ErrID_None\n"
           "      ErrMsg  = \"\"\n"
           "   END IF\n"
           "\n"
           "   IF ( ErrStat /= ErrID_None ) THEN\n")
    neu = ("   ELSE\n"
           "      ErrStat = ErrID_None\n"
           "      ErrMsg  = \"\"\n"
           "   END IF\n"
           "   IF ( PRESENT(LKopie) .AND. tsInfo == 0 .AND. .NOT. tsNur ) LKopie(1:SIZE(TRH)) = TRH(1:SIZE(TRH))\n"
           "\n"
           "   IF ( ErrStat /= ErrID_None ) THEN\n")
    s = ersetze_in_routine(s, C, alt, neu)

    # --- 2. IEC-Routine --------------------------------------------------------
    R = "CalcFourierCoeffs_IEC"
    s = ersetze_in_routine(s, R,
        "INTEGER(4),  ALLOCATABLE      :: tsFrow(:)      ! je Faden: erste Spalte != 0 je Zeile",
        "INTEGER(4),  ALLOCATABLE      :: tsFrow(:)      ! je Faden: erste Spalte != 0 je Zeile\n"
        "LOGICAL                       :: tsAlleGleich   ! u,v,w: IEC mit identischen Parametern -> eine Zerlegung\n"
        "INTEGER                       :: tsIV\n"
        "REAL(ReKi), ALLOCATABLE       :: tsLKopie(:)    ! je Faden: Faktor L fuer die weiteren Komponenten")
    alt = "   DO IVec = 1,3\n   \n      IF (p%met%SCMod(IVec) /= CohMod_IEC) CYCLE"
    neu = ("   tsAlleGleich = ALL( p%met%SCMod == CohMod_IEC ) .AND. &\n"
           "                  p%met%InCDec(1) == p%met%InCDec(2) .AND. p%met%InCDec(2) == p%met%InCDec(3) .AND. &\n"
           "                  p%met%InCohB(1) == p%met%InCohB(2) .AND. p%met%InCohB(2) == p%met%InCohB(3)\n"
           "   IF (tsAlleGleich) CALL WrScr( '    u,v,w: identische Kohaerenzparameter - eine Cholesky je Frequenz' )\n"
           "\n"
           "   DO IVec = 1,3\n   \n      IF (p%met%SCMod(IVec) /= CohMod_IEC) CYCLE")
    s = ersetze_in_routine(s, R, alt, neu)
    s = ersetze_in_routine(s, R,
        "      V(:,:,IVec) = 0.0_ReKi\n",
        "      IF (tsAlleGleich .AND. IVec > 1) CYCLE   ! bereits bei IVec=1 mitgerechnet\n"
        "      V(:,:,IVec) = 0.0_ReKi\n"
        "      IF (tsAlleGleich) V(:,:,2:3) = 0.0_ReKi\n")
    s = ersetze_in_routine(s, R,
        "      !$OMP&  PRIVATE(IFreq, Indx, I, J, TRH_t, tsStat, ErrStat3, ErrMsg3, tsTab, tsFrow, tsIndx0, &",
        "      !$OMP&  PRIVATE(IFreq, Indx, I, J, TRH_t, tsStat, ErrStat3, ErrMsg3, tsTab, tsFrow, tsIndx0, tsLKopie, tsIV, &")
    s = ersetze_in_routine(s, R,
        "      ALLOCATE( TRH_t(SIZE(TRH)), tsTab(2*tsK), tsFrow(tsN), STAT=tsStat )",
        "      ALLOCATE( TRH_t(SIZE(TRH)), tsTab(2*tsK), tsFrow(tsN), tsLKopie(MERGE(SIZE(TRH),1,tsAlleGleich)), STAT=tsStat )")
    s = ersetze_in_routine(s, R,
        "      IF (ALLOCATED(tsFrow)) DEALLOCATE( tsFrow )",
        "      IF (ALLOCATED(tsFrow)) DEALLOCATE( tsFrow )\n"
        "      IF (ALLOCATED(tsLKopie)) DEALLOCATE( tsLKopie )")
    s = ersetze_in_routine(s, R,
        "         CALL Coh2H(    p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3, tsFrow )",
        "         IF (tsAlleGleich) THEN\n"
        "            CALL Coh2H( p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3, tsFrow, LKopie=tsLKopie )\n"
        "         ELSE\n"
        "            CALL Coh2H( p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3, tsFrow )\n"
        "         END IF")
    s = ersetze_in_routine(s, R,
        "         CALL H2Coeffs( IVec, IFreq, TRH_t, PhaseAngles, V, p%grid%NPoints )",
        "         CALL H2Coeffs( IVec, IFreq, TRH_t, PhaseAngles, V, p%grid%NPoints )\n"
        "         IF (tsAlleGleich) THEN\n"
        "            ! v und w: derselbe Faktor L, nur andere Spektren und Phasen\n"
        "            DO tsIV = 2, 3\n"
        "               TRH_t(1:tsNP) = tsLKopie(1:tsNP)\n"
        "               CALL Coh2H( p, tsIV, IFreq, TRH_t, S, ErrStat3, ErrMsg3, tsFrow, NurSkalieren=.TRUE. )\n"
        "               CALL H2Coeffs( tsIV, IFreq, TRH_t, PhaseAngles, V, p%grid%NPoints )\n"
        "            END DO\n"
        "         END IF")

    p.write_text(s)
    print(f"{p.name}: eine Cholesky je Frequenz fuer u,v,w bei identischen Parametern.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

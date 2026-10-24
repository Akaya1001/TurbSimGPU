#!/usr/bin/env python3
"""Ersetzt den Kohaerenzaufbau (exp ueber alle NPacked Paare) durch Dist-Klassen,
glibc-Tabellen und ein Gather — bitgleich (idempotent).

Befund (verifiziert 30.09.2026): Das exp-Argument haengt je Frequenz nur vom
Bitmuster von Dist ab; eine Box hat nur K verschiedene (std 1240, W340 4671,
W960 13 967). Welche exp-Implementierung ein Element bekommt, haengt von seiner
Position ab: gfortran vektorisiert die I-Schleife paarweise mit libmvec
(_ZGVbN2v_exp); das letzte Element I=N jeder Spalte UNGERADER Laenge N-J+1
rechnet skalar mit glibc exp (__exp_fma). Beide weichen bei ~43 % der
Argumente um bis zu 3 ULP ab. Der C-Kern (turbsim/kern/ts_kern.c)
rechnet je Frequenz 2K Werte mit den ECHTEN glibc-Funktionen und verteilt sie
nach Klasse und Paritaet; das Ergebnis ist bitweise dasselbe Feld TRH.

Nebenprodukt: frow(I) = erste Spalte mit Eintrag != 0 je Zeile (fuer die
Huellen-Auslassung der Cholesky in Stufe 1b).

Setzt patch_turbsim_instr.py und patch_turbsim_omp353.py voraus.

Verwendung:  python3 patch_turbsim_gather.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine, routine_grenzen  # noqa: E402

MARKE = "ts_cls_build"

INTERFACE = """   INTEGER(8), SAVE :: tsMess_Rate = 0_8

   ! --- bitgleiche C-Bausteine (turbsim/kern/ts_kern.c) ----------------
   ! Alle Argumente per Referenz; Felder sind die gepackten LAPACK-'L'-Felder.
   INTERFACE
      SUBROUTINE ts_cls_build( np, n, dist, distu, cls, distk, distuk, k ) BIND(C, NAME="ts_cls_build")
         INTEGER(4), INTENT(IN)    :: np, n
         REAL(8),    INTENT(IN)    :: dist(*), distu(*)
         INTEGER(4), INTENT(OUT)   :: cls(*)
         REAL(8),    INTENT(OUT)   :: distk(*), distuk(*)
         INTEGER(4), INTENT(OUT)   :: k
      END SUBROUTINE ts_cls_build
      SUBROUTINE ts_exp_tab( k, distk, distuk, uhub, freq, incdec, incohb, tab ) BIND(C, NAME="ts_exp_tab")
         INTEGER(4), INTENT(IN)    :: k
         REAL(8),    INTENT(IN)    :: distk(*), distuk(*), uhub, freq, incdec, incohb
         REAL(8),    INTENT(OUT)   :: tab(*)
      END SUBROUTINE ts_exp_tab
      SUBROUTINE ts_coh_fill( n, indx0, cls, tab, trh, frow ) BIND(C, NAME="ts_coh_fill")
         INTEGER(4), INTENT(IN)    :: n, indx0, cls(*)
         REAL(8),    INTENT(IN)    :: tab(*)
         REAL(8),    INTENT(INOUT) :: trh(*)
         INTEGER(4), INTENT(OUT)   :: frow(*)
      END SUBROUTINE ts_coh_fill
   END INTERFACE"""


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Gather bereits eingebaut — weiter.")
        return
    assert "tsMess_Coh" in s and "TRH_t" in s, "instr und omp353 muessen zuerst laufen."

    # --- 0. Schnittstellen auf Modulebene (Spezifikationsteil) ----------
    alt = "   INTEGER(8), SAVE :: tsMess_Rate = 0_8"
    assert s.count(alt) == 1
    s = s.replace(alt, INTERFACE, 1)

    R = "CalcFourierCoeffs_IEC"

    # --- 1. Lokale Groessen ----------------------------------------------
    s = ersetze_in_routine(s, R,
        "REAL(ReKi), ALLOCATABLE       :: Dist(:)        ! The distance between points",
        "REAL(ReKi), ALLOCATABLE       :: Dist(:)        ! The distance between points\n"
        "INTEGER(4),  ALLOCATABLE      :: tsCls(:)       ! je gepacktem Eintrag: 2*Klasse + Paritaet (1 = skalares exp)\n"
        "REAL(ReKi), ALLOCATABLE       :: tsDistK(:)     ! Klassenrepraesentanten Dist (aufsteigend)\n"
        "REAL(ReKi), ALLOCATABLE       :: tsDistUK(:)    ! dito DistU\n"
        "INTEGER(4)                    :: tsK            ! Klassenzahl\n"
        "INTEGER(4)                    :: tsNP, tsN, tsIndx0\n"
        "REAL(ReKi), ALLOCATABLE       :: tsTab(:)       ! je Faden: 2K exp-Werte der aktuellen Frequenz\n"
        "INTEGER(4),  ALLOCATABLE      :: tsFrow(:)      ! je Faden: erste Spalte != 0 je Zeile")

    # --- 2. Klassen einmal je Lauf bilden (vor der IVec-Schleife) --------
    alt = "   DO IVec = 1,3\n   \n      IF (p%met%SCMod(IVec) /= CohMod_IEC) CYCLE"
    neu = ("   ! Dist-Klassen: nur K verschiedene Bitmuster; das exp-Argument haengt nur davon ab.\n"
           "   tsNP = p%grid%NPacked\n"
           "   tsN  = p%grid%NPoints\n"
           "   ALLOCATE( tsCls(tsNP), tsDistK(tsNP), tsDistUK(tsNP), STAT=ErrStat2 )\n"
           "   IF (ErrStat2 /= 0) THEN\n"
           "      CALL SetErrStat( ErrID_Fatal, 'Dist-Klassen konnten nicht angelegt werden.', ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC' )\n"
           "      CALL Cleanup()\n"
           "      RETURN\n"
           "   END IF\n"
           "   CALL ts_cls_build( tsNP, tsN, Dist, DistU, tsCls, tsDistK, tsDistUK, tsK )\n"
           "   IF (tsK <= 0) THEN\n"
           "      CALL SetErrStat( ErrID_Fatal, 'Dist-Klassen: ts_cls_build fehlgeschlagen.', ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC' )\n"
           "      CALL Cleanup()\n"
           "      RETURN\n"
           "   END IF\n"
           "   CALL WrScr( '    Dist-Klassen: '//TRIM(Num2LStr(tsK))//' von '//TRIM(Num2LStr(tsNP))//' Eintraegen' )\n"
           "\n"
           "   DO IVec = 1,3\n   \n      IF (p%met%SCMod(IVec) /= CohMod_IEC) CYCLE")
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 3. Fadenprivate Puffer -------------------------------------------
    s = ersetze_in_routine(s, R,
        "      !$OMP&  PRIVATE(IFreq, Indx, I, J, TRH_t, tsStat, ErrStat3, ErrMsg3, &",
        "      !$OMP&  PRIVATE(IFreq, Indx, I, J, TRH_t, tsStat, ErrStat3, ErrMsg3, tsTab, tsFrow, tsIndx0, &")
    s = ersetze_in_routine(s, R,
        "      ALLOCATE( TRH_t(SIZE(TRH)), STAT=tsStat )",
        "      ALLOCATE( TRH_t(SIZE(TRH)), tsTab(2*tsK), tsFrow(tsN), STAT=tsStat )")
    s = ersetze_in_routine(s, R,
        "      IF (ALLOCATED(TRH_t)) DEALLOCATE( TRH_t )",
        "      IF (ALLOCATED(TRH_t)) DEALLOCATE( TRH_t )\n"
        "      IF (ALLOCATED(tsTab)) DEALLOCATE( tsTab )\n"
        "      IF (ALLOCATED(tsFrow)) DEALLOCATE( tsFrow )")

    # --- 4. Die exp-Doppelschleife durch Tabelle + Gather ersetzen --------
    a, e = routine_grenzen(s, R)
    koerper = s[a:e]
    start_marke = "         DO J=max(1, p%usr%NPoints),p%grid%NPoints\n"
    ende_marke = "         ENDDO ! J\n"
    i0 = koerper.find(start_marke)
    assert i0 >= 0 and koerper.find(start_marke, i0 + 1) < 0, "Schleifenkopf nicht eindeutig"
    i1 = koerper.find(ende_marke, i0)
    assert i1 >= 0
    block = koerper[i0:i1 + len(ende_marke)]
    assert block.count("TRH_t(Indx) = EXP(") == 1, "exp-Zuweisung nicht gefunden (omp353 zuerst?)"
    neu = ("         ! Kohaerenz je Klasse mit den echten glibc-Funktionen, dann Gather nach\n"
           "         ! Klasse und Paritaet (Vektor-/Skalar-exp wie im Referenz-Binary).\n"
           "         tsIndx0 = Indx\n"
           "         CALL ts_exp_tab( tsK, tsDistK, tsDistUK, p%UHub, p%grid%Freq(IFreq), &\n"
           "                          p%met%InCDec(IVec), p%met%InCohB(IVec), tsTab )\n"
           "         CALL ts_coh_fill( tsN, tsIndx0, tsCls, tsTab, TRH_t, tsFrow )\n"
           "         Indx = tsNP + 1\n")
    koerper = koerper[:i0] + neu + koerper[i1 + len(ende_marke):]
    s = s[:a] + koerper + s[e:]

    # --- 5. Aufraeumen -----------------------------------------------------
    s = ersetze_in_routine(s, R,
        "      IF ( ALLOCATED( DistU     ) ) DEALLOCATE( DistU     )",
        "      IF ( ALLOCATED( DistU     ) ) DEALLOCATE( DistU     )\n"
        "      IF ( ALLOCATED( tsCls     ) ) DEALLOCATE( tsCls     )\n"
        "      IF ( ALLOCATED( tsDistK   ) ) DEALLOCATE( tsDistK   )\n"
        "      IF ( ALLOCATED( tsDistUK  ) ) DEALLOCATE( tsDistUK  )")

    p.write_text(s)
    print(f"{p.name}: Kohaerenzaufbau durch Klassen-Tabelle + Gather ersetzt.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

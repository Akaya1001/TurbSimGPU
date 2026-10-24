#!/usr/bin/env python3
"""Parallelisiert TurbSims IEC-Frequenzschleife mit OpenMP — Fassung fuer den
v3.5.3-Referenzbaum (idempotent).

Unterschiede zu patch_turbsim_omp.py (v3.4.1-Baum):
  * Kein Chol_Blocked (der v3.5.3-Baum behaelt LAPACK_pptrf bzw. bekommt
    spaeter ts_chol_spec): Schritt 6 entfaellt.
  * Der Identitaetspfad (v, w) wird NICHT hier parallelisiert, sondern von
    patch_turbsim_eye.py in geschlossener Form O(N) ersetzt.
  * Fehlerpfad deterministisch: Die Referenz bricht bei der ERSTEN
    scheiternden Frequenz ab und meldet genau diese. Mehrere Faeden koennen
    gleichzeitig scheitern; darum wird unter CRITICAL nur die kleinste
    Frequenz mit ihrer Meldung behalten und erst NACH der Schleife einmal
    per SetErrStat gemeldet. Damit ist die Fehlermeldung bitgleich zur
    seriellen Referenz, unabhaengig vom Faden-Timing.
  * SCHEDULE(DYNAMIC,tsChunk), tsChunk = MAX(1, MIN(64, NumFreq/48)): grosse Chunks vermeiden
    False Sharing auf V, bei wenigen Frequenzen bleibt kein Kern leer. Bitgleichheit haengt nicht
    am Schedule.

Warum das bitgleich bleibt: Die Frequenzen sind vollstaendig unabhaengig.
H2Coeffs schreibt nach V(2*IFreq-1, :, IVec) und V(2*IFreq, :, IVec) —
je Frequenz eine eigene Scheibe. Innerhalb einer Frequenz aendert sich
keine Rechenreihenfolge; es aendert sich nur, WER rechnet.

Setzt patch_turbsim_instr.py voraus (Zeitstempel muessen lokal sein).

Verwendung:  python3 patch_turbsim_omp353.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "TRH_t"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Frequenzschleife bereits parallelisiert — weiter.")
        return
    assert "tsMess_Coh" in s, (
        "patch_turbsim_instr.py muss zuerst laufen (Zeitstempel muessen lokal sein)."
    )
    assert "Chol_Blocked" not in s, "omp353 ist fuer den v3.5.3-Baum ohne Chol_Blocked gedacht."

    R = "CalcFourierCoeffs_IEC"

    # --- 1. Zusaetzliche lokale Groessen --------------------------------
    s = ersetze_in_routine(s, R,
        "INTEGER(8)                    :: tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3",
        "INTEGER(8)                    :: tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3\n"
        "REAL(ReKi), ALLOCATABLE       :: TRH_t(:)      ! Kohaerenzmatrix je Faden\n"
        "INTEGER(IntKi)                :: ErrStat3\n"
        "CHARACTER(MaxMsgLen)          :: ErrMsg3\n"
        "LOGICAL                       :: tsFehl        ! Speicher je Faden fehlgeschlagen\n"
        "LOGICAL                       :: tsAbbruch     ! eine Frequenz ist fatal gescheitert\n"
        "INTEGER                       :: tsStat\n"
        "INTEGER(IntKi)                :: tsFehlIFreq   ! kleinste gescheiterte Frequenz (wie die serielle Referenz)\n"
        "INTEGER(IntKi)                :: tsFehlStat\n"
        "CHARACTER(MaxMsgLen)          :: tsFehlMsg\n"
        "INTEGER                       :: tsChunk       ! OpenMP-Chunk, an NumFreq angepasst")

    # --- 2. Die Schleife selbst -----------------------------------------
    alt = """      DO IFREQ = 1,p%grid%NumFreq
         CALL SYSTEM_CLOCK( tsMess_c0, tsMess_Rate )"""
    neu = """      ! Taktrate einmal vor dem parallelen Bereich holen: sonst schreiben
      ! alle Faeden dieselbe Modulvariable (harmlos, aber ein Datenrennen).
      CALL SYSTEM_CLOCK( COUNT_RATE=tsMess_Rate )
      tsFehl      = .FALSE.
      tsAbbruch   = .FALSE.
      tsFehlIFreq = HUGE(1_IntKi)
      tsFehlStat  = ErrID_None
      tsFehlMsg   = ""
      tsChunk     = MAX(1, MIN(64, p%grid%NumFreq/48))

      !$OMP PARALLEL DEFAULT(SHARED) &
      !$OMP&  PRIVATE(IFreq, Indx, I, J, TRH_t, tsStat, ErrStat3, ErrMsg3, &
      !$OMP&          tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3)

      ! Eigener Arbeitsspeicher je Faden, einmal je Faden — nicht je Frequenz.
      ALLOCATE( TRH_t(SIZE(TRH)), STAT=tsStat )
      IF (tsStat /= 0) THEN
         !$OMP CRITICAL (tsFehlerBereich)
         tsFehl = .TRUE.
         !$OMP END CRITICAL (tsFehlerBereich)
      END IF

      ! Chunk bis 64: benachbarte Frequenzen teilen sich Cache-Zeilen in V(2f-1,I)/V(2f,I),
      ! kleine Chunks erzeugen False Sharing. Bei wenigen Frequenzen (Farm-Low 42, kurze
      ! Farm-High 374) wuerden 64er-Chunks Kerne leer lassen: darum NumFreq/48 (12 Faeden x 4).
      !$OMP DO SCHEDULE(DYNAMIC,tsChunk)
      DO IFREQ = 1,p%grid%NumFreq
         ! Nach einem Abbruch die restlichen Frequenzen nur noch durchlaufen —
         ! aus einer OpenMP-Schleife darf man nicht herausspringen.
         IF (tsFehl .OR. tsAbbruch) CYCLE
         CALL SYSTEM_CLOCK( tsMess_c0 )"""
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 3. Aufrufblock: TRH -> TRH_t, Abbruch ueber Kennzeichen ---------
    alt = """         CALL SYSTEM_CLOCK( tsMess_c1 )
         CALL Coh2H(    p, IVec, IFreq, TRH, S, ErrStat2, ErrMsg2 )       
            CALL SetErrStat(ErrStat2, ErrMsg2, ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC')
            IF (ErrStat >= AbortErrLev) THEN
               CALL Cleanup()
               RETURN
            END IF
         CALL SYSTEM_CLOCK( tsMess_c2 )
         CALL H2Coeffs( IVec, IFreq, TRH, PhaseAngles, V, p%grid%NPoints )"""
    neu = """         CALL SYSTEM_CLOCK( tsMess_c1 )
         CALL Coh2H(    p, IVec, IFreq, TRH_t, S, ErrStat3, ErrMsg3 )
         IF (ErrStat3 /= ErrID_None) THEN
            ! Nur die kleinste gescheiterte Frequenz behalten: das ist die, an
            ! der die serielle Referenz abgebrochen haette.
            !$OMP CRITICAL (tsFehlerBereich)
            IF (IFreq < tsFehlIFreq) THEN
               tsFehlIFreq = IFreq
               tsFehlStat  = ErrStat3
               tsFehlMsg   = ErrMsg3
            END IF
            IF (ErrStat3 >= AbortErrLev) tsAbbruch = .TRUE.
            !$OMP END CRITICAL (tsFehlerBereich)
            IF (ErrStat3 >= AbortErrLev) CYCLE
         END IF
         CALL SYSTEM_CLOCK( tsMess_c2 )
         CALL H2Coeffs( IVec, IFreq, TRH_t, PhaseAngles, V, p%grid%NPoints )"""
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 4. Schleifenende + Auswertung des Abbruchs ----------------------
    alt = """         tsMess_H2C  = tsMess_H2C  + (tsMess_c3 - tsMess_c2)
      END DO !IFreq"""
    neu = """         tsMess_H2C  = tsMess_H2C  + (tsMess_c3 - tsMess_c2)
      END DO !IFreq
      !$OMP END DO
      IF (ALLOCATED(TRH_t)) DEALLOCATE( TRH_t )
      !$OMP END PARALLEL

      IF (tsFehl) THEN
         CALL SetErrStat( ErrID_Fatal, 'Kohaerenzmatrix je Faden konnte nicht'// &
              ' angelegt werden - OMP_NUM_THREADS senken.', ErrStat, ErrMsg, &
              'CalcFourierCoeffs_IEC' )
      END IF
      IF (tsFehlIFreq /= HUGE(1_IntKi)) THEN
         CALL SetErrStat(tsFehlStat, tsFehlMsg, ErrStat, ErrMsg, 'CalcFourierCoeffs_IEC')
      END IF
      IF (ErrStat >= AbortErrLev) THEN
         CALL Cleanup()
         RETURN
      END IF"""
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 5. Die inneren TRH-Zugriffe auf den Fadenspeicher umbiegen ------
    for a, n in (
        ("            TRH(Indx) =  1.0_ReKi", "            TRH_t(Indx) =  1.0_ReKi"),
        ("               TRH(Indx) = 0.0_ReKi", "               TRH_t(Indx) = 0.0_ReKi"),
        ("                  TRH(Indx) = EXP( -1.0_ReKi * p%met%InCDec(IVec) * &",
         "                  TRH_t(Indx) = EXP( -1.0_ReKi * p%met%InCDec(IVec) * &"),
    ):
        s = ersetze_in_routine(s, R, a, n)

    p.write_text(s)
    print(f"{p.name}: Frequenzschleife parallelisiert (omp353).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

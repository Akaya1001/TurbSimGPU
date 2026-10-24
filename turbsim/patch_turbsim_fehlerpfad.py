#!/usr/bin/env python3
"""Fehlerpfad der parallelen Frequenzschleife: immer die kleinste scheiternde Frequenz melden (idempotent).

omp353 brach nach dem ersten fatalen Fehler ALLE uebrigen Frequenzen ab (`IF (tsFehl .OR. tsAbbruch)
CYCLE`), auch kleinere, die ein anderer Faden noch nicht erreicht hatte. Scheitern mehrere Frequenzen,
haengt die gemeldete dann vom Timing ab — die serielle Referenz meldet aber die kleinste scheiternde
Frequenz ueberhaupt. Neu: nach einem Abbruch werden nur noch Frequenzen OBERHALB der kleinsten fatal
gescheiterten uebersprungen; alle darunter laufen weiter (und koennen ihrerseits scheitern). Die
Grenze wird atomar gelesen und unter dem kritischen Abschnitt atomar geschrieben.

Gefunden in der Pruefung zum GPU-Plan (14.10.2026). Im Betrieb scheitert praktisch keine Zerlegung
(Kohaerenzmatrizen sind positiv definit); es geht um einen deterministischen Fehlerpfad, auch fuer die
spaetere Frequenzteilung zwischen CPU und GPU.

Setzt patch_turbsim_omp353.py voraus (laeuft zuletzt in der Kette).

Verwendung:  python3 patch_turbsim_fehlerpfad.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsAbbIFreq"
R = "CalcFourierCoeffs_IEC"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: Fehlerpfad bereits deterministisch — weiter.")
        return
    assert "tsFehlIFreq" in s, "patch_turbsim_omp353.py muss zuerst laufen."

    s = ersetze_in_routine(s, R,
        "INTEGER(IntKi)                :: tsFehlIFreq   ! kleinste gescheiterte Frequenz (wie die serielle Referenz)\n",
        "INTEGER(IntKi)                :: tsFehlIFreq   ! kleinste gescheiterte Frequenz (wie die serielle Referenz)\n"
        "INTEGER(IntKi)                :: tsAbbIFreq    ! kleinste FATAL gescheiterte Frequenz\n"
        "INTEGER(IntKi)                :: tsGrenze      ! deren Kopie je Faden\n")
    s = ersetze_in_routine(s, R,
        "      tsFehlIFreq = HUGE(1_IntKi)\n",
        "      tsFehlIFreq = HUGE(1_IntKi)\n"
        "      tsAbbIFreq  = HUGE(1_IntKi)\n")
    s = ersetze_in_routine(s, R,
        "      !$OMP&          tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3)\n",
        "      !$OMP&          tsMess_c0, tsMess_c1, tsMess_c2, tsMess_c3, tsGrenze)\n")
    s = ersetze_in_routine(s, R,
        "         ! Nach einem Abbruch die restlichen Frequenzen nur noch durchlaufen —\n"
        "         ! aus einer OpenMP-Schleife darf man nicht herausspringen.\n"
        "         IF (tsFehl .OR. tsAbbruch) CYCLE\n",
        "         ! Nach einem Abbruch nur noch Frequenzen UNTER der kleinsten fatal gescheiterten rechnen\n"
        "         ! (sie koennten selbst scheitern): die serielle Referenz meldet die kleinste scheiternde\n"
        "         ! Frequenz ueberhaupt. Aus einer OpenMP-Schleife darf man nicht herausspringen.\n"
        "         IF (tsFehl) CYCLE\n"
        "         !$OMP ATOMIC READ\n"
        "         tsGrenze = tsAbbIFreq\n"
        "         IF (IFreq > tsGrenze) CYCLE\n")
    s = ersetze_in_routine(s, R,
        "            IF (ErrStat3 >= AbortErrLev) tsAbbruch = .TRUE.\n"
        "            !$OMP END CRITICAL (tsFehlerBereich)\n",
        "            IF (ErrStat3 >= AbortErrLev) THEN\n"
        "               tsAbbruch = .TRUE.\n"
        "               IF (IFreq < tsAbbIFreq) THEN\n"
        "                  !$OMP ATOMIC WRITE\n"
        "                  tsAbbIFreq = IFreq\n"
        "               END IF\n"
        "            END IF\n"
        "            !$OMP END CRITICAL (tsFehlerBereich)\n")
    p.write_text(s)
    print(f"{p.name}: Fehlerpfad deterministisch (kleinste scheiternde Frequenz).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

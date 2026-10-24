#!/usr/bin/env python3
"""H2Coeffs zeilenblockweise in C (ts_h2c), bitgleich (idempotent).

Die Referenz macht je Frequenz N^2/2 verstreute Lese-Modifizier-Schreib-Zugriffe auf
V (Schrittweite NumSteps*8 B); bei der vollen Box (V = 5-18 GB) kostet das zehnmal
mehr als bei der 120-s-Fassung (gemessen: H2Coeffs 275 s Fadenzeit bei std, 738 s
bei W960). ts_h2c haelt die V-Werte eines Zeilenblocks im L1 und liest TRH einmal
sequenziell. Je Element bleibt die Summe in aufsteigendem J mit getrennter
Multiplikation und Addition — die Rechenreihenfolge der Referenz.

COS/SIN der Phasen rechnet der Kern selbst, skalar mit glibc sincos wie die Referenz
(eine Fortran-Schleife ueber COS/SIN vektorisiert gfortran mit libmvec: andere Bits —
gemessen: 19 % der V-Worte um einige ULP anders, .bts dennoch gleich).

Setzt patch_turbsim_gather.py voraus (INTERFACE-Block).

Verwendung:  python3 patch_turbsim_h2c.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "ts_h2c"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: H2Coeffs bereits in C — weiter.")
        return
    assert "ts_coh_fill" in s and s.count("   END INTERFACE") == 1, "patch_turbsim_gather.py muss zuerst laufen (INTERFACE-Block)."
    alt = "   END INTERFACE"
    assert s.count(alt) == 1
    s = s.replace(alt,
        "      SUBROUTINE ts_h2c( n, trh, phase, ldp, nfp, ifreq, v, ldv, npv, if1, ivec ) BIND(C, NAME=\"ts_h2c\")\n"
        "         INTEGER(4), INTENT(IN)    :: n, ldp, nfp, ifreq, ldv, npv, if1, ivec\n"
        "         REAL(8),    INTENT(IN)    :: trh(*), phase(*)\n"
        "         REAL(8),    INTENT(INOUT) :: v(*)\n"
        "      END SUBROUTINE ts_h2c\n"
        "   END INTERFACE", 1)

    H = "H2Coeffs"
    s = ersetze_in_routine(s, H,
        "REAL(ReKi)                       :: CPh                                      ! Cosine of the random phase\n"
        "REAL(ReKi)                       :: SPh                                      ! Sine of the random phase\n",
        "REAL(ReKi)                       :: CPh                                      ! (unbenutzt; sincos rechnet ts_h2c)\n"
        "REAL(ReKi)                       :: SPh                                      ! (unbenutzt)\n"
        "INTEGER(4)                       :: tsLdv, tsNpv, tsLdp, tsNfp\n")
    alt = """   DO J=1,NPoints

         ! Apply a random phase to each of the columns of H to
         ! produce random phases in the wind component.
         ! Then sum each of the rows into the vector V.
         
      CPh   = COS( PhaseAngles(J,IFreq,IVec) )
      SPh   = SIN( PhaseAngles(J,IFreq,IVec) )

      Indx = NPoints*(J-1) - J*(J-1)/2 + J !Index of H(J,J)
      DO I=J,NPoints

         V(IF1,I,IVec) = V(IF1,I,IVec) + TRH(Indx)*CPh  !Real part
         V(IF2,I,IVec) = V(IF2,I,IVec) + TRH(Indx)*SPh  !Imaginary part

         Indx = Indx + 1      !H(I,J)

      ENDDO ! I
   ENDDO ! J
"""
    neu = """   ! Zeilensumme zeilenblockweise in C: je Element dieselbe Reihenfolge (J aufsteigend,
   ! mul dann add), V einmal gelesen/geschrieben statt N^2/2 verstreuter Zugriffe.
   ! COS/SIN rechnet der Kern skalar mit glibc sincos — eine Fortran-Schleife ueber
   ! COS/SIN wuerde gfortran mit libmvec vektorisieren (andere Bits).
   tsLdv = SIZE(V,1)
   tsNpv = SIZE(V,2)
   tsLdp = SIZE(PhaseAngles,1)
   tsNfp = SIZE(PhaseAngles,2)
   CALL ts_h2c( NPoints, TRH, PhaseAngles, tsLdp, tsNfp, IFreq, V, tsLdv, tsNpv, IF1, IVec )
"""
    s = ersetze_in_routine(s, H, alt, neu)
    p.write_text(s)
    print(f"{p.name}: H2Coeffs zeilenblockweise in C (ts_h2c).")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

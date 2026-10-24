#!/usr/bin/env python3
"""Serieller Rest nach der Frequenzschleife: IFFT und Mittelwind-Rotation
punktweise parallel — bitgleich (idempotent).

1. Coeffs2TimeSeries: je (Komponente, Punkt) eine unabhaengige float32-IFFT
   (FFTPACK RFFTB). RFFTB benutzt WSAVE als ARBEITSFELD (fftpack4.1.f:156-161),
   darum bekommt jeder Faden eigene FFT_Data (InitFFT je Faden) und eigenes Work.
   Die Rundung double->float32 je Element und die Ruecksortierung sind Kopien;
   die FFT je Punkt ist dieselbe Rechnung wie zuvor -> bitgleich.
2. AddMeanAndRotate: der Aufruf CalculateWindComponents(v3, ..., V(ITime,IPoint,:))
   uebergibt einen nicht zusammenhaengenden Abschnitt als Ausgabe; gfortran legt
   dafuer je Aufruf ein Temporaerfeld an (malloc/free, N*NumSteps-mal). Neu: lokales
   vo(3) als Ausgabe, danach Zuweisung (Kopie = exakt); OMP ueber IPoint.

Verwendung:  python3 patch_turbsim_rest.py <baum>/modules/turbsim/src/TSsubs.f90
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ofpatch import ersetze_in_routine  # noqa: E402

MARKE = "tsRestFFT"


def patch(p: Path) -> None:
    s = p.read_text()
    if MARKE in s:
        print(f"{p.name}: serieller Rest bereits parallelisiert — weiter.")
        return

    # --- 1. Coeffs2TimeSeries -------------------------------------------------
    R = "Coeffs2TimeSeries"
    s = ersetze_in_routine(s, R,
        "   INTEGER(IntKi)                   :: ErrStat2                      ! Error level (local)",
        "   INTEGER(IntKi)                   :: ErrStat2                      ! Error level (local)\n"
        "   TYPE(FFT_DataType)               :: tsRestFFT                     ! FFT-Arbeitsdaten je Faden (WSAVE ist Arbeitsfeld!)\n"
        "   REAL(SiKi), ALLOCATABLE          :: tsWork(:)                     ! Koeffizienten je Faden\n"
        "   INTEGER(IntKi)                   :: tsStat2, tsStatFFT")

    alt = """DO IVec=1,3

   CALL WrScr ( '    '//Comp(IVec)//'-component' )

   DO IPoint=1,NPoints    !NTotB

         ! Overwrite the first point with zero.  This sets the real (and 
         ! imaginary) part of the steady-state value to zero so that we 
         ! can add in the mean value later.

      Work(1) = 0.0_ReKi

!      DO ITime = 2,NumSteps-1
      DO ITime = 2,NumSteps
         Work(ITime) = V(ITime-1, IPoint, IVec)
      ENDDO ! ITime

      IF (iPoint > NUsrPoints) THEN
         ! BJJ: we can't override this for the user-input spectra or we don't get the correct time series out.
         ! Per JMJ, I will keep this here for the other points, but I personally think it could be skipped, too.
         
         ! Now, let's add a complex zero to the end to set the power in the Nyquist
         ! frequency to zero.
         
         Work(NumSteps) = 0.0
      END IF
      
      

         ! perform FFT

      CALL ApplyFFT( Work, FFT_Data, ErrStat2 )
         IF (ErrStat2 /= ErrID_None ) THEN
            CALL SetErrStat(ErrStat2, 'Error in ApplyFFT for point '//TRIM(Num2LStr(IPoint))//'.', ErrStat, ErrMsg, 'Coeffs2TimeSeries' )
            IF (ErrStat >= AbortErrLev) EXIT
         END IF
        
      V(:,IPoint,IVec) = Work

   ENDDO ! IPoint

ENDDO ! IVec """
    neu = """tsStatFFT = 0
DO IVec=1,3

   CALL WrScr ( '    '//Comp(IVec)//'-component' )

   ! Punkte sind unabhaengig; je Faden eigene FFT-Daten (RFFTB schreibt in WSAVE) und eigenes Work.
   !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(IPoint, ITime, tsRestFFT, tsWork, tsStat2)
   ALLOCATE( tsWork(NumSteps) )
   CALL InitFFT( NumSteps, tsRestFFT, ErrStat=tsStat2 )
   IF (tsStat2 /= ErrID_None) THEN
      !$OMP ATOMIC WRITE
      tsStatFFT = tsStat2
   END IF
   !$OMP DO SCHEDULE(STATIC)
   DO IPoint=1,NPoints    !NTotB
      IF (tsStatFFT /= 0) CYCLE

         ! Overwrite the first point with zero.  This sets the real (and 
         ! imaginary) part of the steady-state value to zero so that we 
         ! can add in the mean value later.

      tsWork(1) = 0.0_ReKi

      DO ITime = 2,NumSteps
         tsWork(ITime) = V(ITime-1, IPoint, IVec)
      ENDDO ! ITime

      IF (iPoint > NUsrPoints) THEN
         ! Now, let's add a complex zero to the end to set the power in the Nyquist
         ! frequency to zero.
         tsWork(NumSteps) = 0.0
      END IF

         ! perform FFT

      CALL ApplyFFT( tsWork, tsRestFFT, tsStat2 )
         IF (tsStat2 /= ErrID_None ) THEN
            !$OMP ATOMIC WRITE
            tsStatFFT = tsStat2
         END IF
        
      V(:,IPoint,IVec) = tsWork

   ENDDO ! IPoint
   !$OMP END DO
   CALL ExitFFT( tsRestFFT, tsStat2 )
   DEALLOCATE( tsWork )
   !$OMP END PARALLEL
   IF (tsStatFFT /= 0) THEN
      CALL SetErrStat(tsStatFFT, 'Error in ApplyFFT.', ErrStat, ErrMsg, 'Coeffs2TimeSeries' )
      IF (ErrStat >= AbortErrLev) EXIT
   END IF

ENDDO ! IVec """
    s = ersetze_in_routine(s, R, alt, neu)

    # --- 2. AddMeanAndRotate --------------------------------------------------
    A = "AddMeanAndRotate"
    s = ersetze_in_routine(s, A,
        "   REAL(ReKi)                                  ::  v3(3)             ! temporary 3-component array containing velocity",
        "   REAL(ReKi)                                  ::  v3(3)             ! temporary 3-component array containing velocity\n"
        "   REAL(ReKi)                                  ::  vo(3)             ! Ausgabe je (Zeit, Punkt): kein Temporaerfeld je Aufruf")
    alt = """   DO IPoint=1,p%grid%Npoints   
      DO ITime=1,p%grid%NumSteps

            ! Add mean wind speed to the streamwise component and
            ! Rotate the wind to the X-Y-Z (inertial) reference frame coordinates:
            
         v3 = V(ITime,IPoint,:) 
         CALL CalculateWindComponents( v3, U(IPoint), HWindDir(IPoint), VWindDir(IPoint), V(ITime,IPoint,:) )
                        
      ENDDO ! ITime
         
   ENDDO ! IPoint   """
    neu = """   ! Punkte und Zeitschritte sind unabhaengig; vo(3) statt des nicht zusammenhaengenden
   ! Abschnitts als Ausgabe spart gfortran je Aufruf ein Temporaerfeld (malloc/free).
   !$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(IPoint, ITime, v3, vo) SCHEDULE(STATIC)
   DO IPoint=1,p%grid%Npoints   
      DO ITime=1,p%grid%NumSteps

            ! Add mean wind speed to the streamwise component and
            ! Rotate the wind to the X-Y-Z (inertial) reference frame coordinates:
            
         v3 = V(ITime,IPoint,:) 
         CALL CalculateWindComponents( v3, U(IPoint), HWindDir(IPoint), VWindDir(IPoint), vo )
         V(ITime,IPoint,:) = vo
                        
      ENDDO ! ITime
         
   ENDDO ! IPoint   
   !$OMP END PARALLEL DO"""
    s = ersetze_in_routine(s, A, alt, neu)

    p.write_text(s)
    print(f"{p.name}: IFFT und AddMeanAndRotate punktweise parallel.")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))

/* ts_h2c.c — H2Coeffs (Veers' Zeilensumme) zeilenblockweise, bitgleich.
 *
 * Referenz (TSsubs.f90 H2Coeffs): J aussen, I innen:
 *    V(IF1,I) = V(IF1,I) + TRH(I,J)*CPh(J);  V(IF2,I) = V(IF2,I) + TRH(I,J)*SPh(J)
 * Je Zeile I ist das eine Summe in aufsteigendem J, Multiplikation und Addition
 * getrennt gerundet (kein FMA). Diese Reihenfolge je Element bleibt hier exakt
 * erhalten; nur die Schleifen sind geblockt: ein Zeilenblock der V-Werte liegt im
 * L1, TRH wird einmal sequenziell gelesen, V einmal geschrieben — statt N^2/2
 * verstreuter Lese-Modifizier-Schreib-Zugriffe auf V (Schrittweite NumSteps*8 B).
 *
 * COS/SIN der Phasen: die Referenz ruft je Spalte glibc sincos (skalar). Eine
 * Fortran-Schleife ueber COS/SIN vektorisiert gfortran mit libmvec (_ZGVbN2v_cos/sin,
 * bis zu einige ULP anders) — darum rechnet dieser Kern sincos selbst, skalar.
 *
 * Uebersetzen OHNE -mfma, mit -ffp-contract=off (keine Kontraktion) und
 * -fno-builtin-sincos/-sin/-cos, gern -O3 -mavx2 (vmulpd/vaddpd). V ist das ganze
 * Fortran-Feld V(NumSteps,NPoints,3), PhaseAngles das ganze Feld (NPoints,NumFreq,3).
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#define RB 256

static inline size_t col0(int32_t n, int32_t j) { return (size_t)j * (size_t)n - ((size_t)j * (size_t)(j - 1)) / 2; }   /* 0-basiert */

/* Fortran bind(C): ts_h2c(n, trh, phase, ldp, nfp, ifreq, v, ldv, npv, if1, ivec)
 *   trh      : gepackt 'L' (NPacked)
 *   phase    : PhaseAngles(ldp, nfp, 3); Spalte J hat Phase phase[(J-1) + ldp*((ifreq-1) + nfp*(ivec-1))]
 *   v        : V(ldv, npv, 3); if1, ifreq, ivec 1-basiert */
void ts_h2c(const int32_t *n_, const double *trh, const double *phase, const int32_t *ldp_, const int32_t *nfp_,
            const int32_t *ifreq_, double *v, const int32_t *ldv_, const int32_t *npv_, const int32_t *if1_, const int32_t *ivec_)
{
   const int32_t n = *n_;
   const size_t ldv = (size_t)*ldv_, npv = (size_t)*npv_;
   double cph[8192], sph[8192];                               /* N <= 8192 Punkte; sonst Rueckfall unten */
   double *cp = cph, *sp = sph;
   if (n > 8192) { cp = (double *)malloc(sizeof(double) * 2 * (size_t)n); sp = cp + n; }
   {
      const double *ph = phase + (size_t)*ldp_ * (((size_t)*ifreq_ - 1) + (size_t)*nfp_ * ((size_t)*ivec_ - 1));
      for (int32_t j = 0; j < n; ++j) sincos(ph[j], &sp[j], &cp[j]);   /* glibc sincos wie die Referenz (H2Coeffs: COS, SIN) */
   }
   double *vb = v + ((size_t)*if1_ - 1) + ldv * npv * ((size_t)*ivec_ - 1);   /* vb[ldv*i] = V(IF1,i+1,IVec), vb[ldv*i+1] = V(IF2,...) */
   double ar[RB], ai[RB];
   for (int32_t i0 = 0; i0 < n; i0 += RB) {
      const int32_t i1 = (i0 + RB < n) ? i0 + RB : n, m = i1 - i0;
      for (int32_t t = 0; t < m; ++t) { ar[t] = vb[ldv * (size_t)(i0 + t)]; ai[t] = vb[ldv * (size_t)(i0 + t) + 1]; }
      for (int32_t j = 0; j < i1; ++j) {                      /* Spalten J <= I fuer alle Zeilen des Blocks */
         const int32_t s = (j > i0) ? j : i0;                 /* erste Zeile der Spalte im Block */
         const double *te = trh + col0(n, j) + (size_t)(s - j);
         const double c = cp[j], sn = sp[j];
         double *pr = ar + (s - i0), *pi = ai + (s - i0);
         const int32_t len = i1 - s;
         for (int32_t t = 0; t < len; ++t) {
            const double x = te[t];
            pr[t] = pr[t] + x * c;                              /* mul, dann add: zwei Rundungen wie die Referenz */
            pi[t] = pi[t] + x * sn;
         }
      }
      for (int32_t t = 0; t < m; ++t) { vb[ldv * (size_t)(i0 + t)] = ar[t]; vb[ldv * (size_t)(i0 + t) + 1] = ai[t]; }
   }
   if (cp != cph) free(cp);
}

/* Identitaetspfad (SCMod = NONE): H ist diagonal, die Zeilensumme von H2Coeffs ist
 *   V(IF1,I) = (+0) + SQRT(ABS(S(IFreq,I)))*COS(Phase);  V(IF2,I) = (+0) + SQRT(ABS(S))*SIN(Phase)
 * (Beweis: alle Nebendiagonalbeitraege sind +-0, (+0)+(+-0) = +0.) sqrt korrekt gerundet,
 * sincos von glibc — skalar, ohne libmvec. S ist das ganze Feld S(NumFreq,NPoints,3).
 * Fortran bind(C): ts_eye(n, s, nfs, nps, phase, ldp, nfp, ifreq, v, ldv, npv, if1, ivec) */
void ts_eye(const int32_t *n_, const double *S, const int32_t *nfs_, const int32_t *nps_,
            const double *phase, const int32_t *ldp_, const int32_t *nfp_, const int32_t *ifreq_,
            double *v, const int32_t *ldv_, const int32_t *npv_, const int32_t *if1_, const int32_t *ivec_)
{
   const int32_t n = *n_;
   const size_t nfs = (size_t)*nfs_, nps = (size_t)*nps_, ldv = (size_t)*ldv_, npv = (size_t)*npv_;
   const size_t f = (size_t)*ifreq_ - 1, c = (size_t)*ivec_ - 1;
   const double *sp = S + f + nfs * nps * c;                                   /* sp[nfs*i] = S(IFreq, i+1, IVec) */
   const double *ph = phase + (size_t)*ldp_ * (f + (size_t)*nfp_ * c);         /* ph[i]     = PhaseAngles(i+1, IFreq, IVec) */
   double *vb = v + ((size_t)*if1_ - 1) + ldv * npv * c;
   for (int32_t i = 0; i < n; ++i) {
      double sn, cs;
      sincos(ph[i], &sn, &cs);
      const double h = sqrt(fabs(sp[nfs * (size_t)i]));
      vb[ldv * (size_t)i]     = vb[ldv * (size_t)i]     + h * cs;
      vb[ldv * (size_t)i + 1] = vb[ldv * (size_t)i + 1] + h * sn;
   }
}

/* ts_gpu_prep.c — CPU-Vorbereitung je Stapel fuer den GPU-Pfad der Frequenzschleife (Stufe G3, Plan 14.10.2026).
 *
 * Die GPU rechnet Gather, Cholesky, S-Skalierung und H2Coeffs. Transzendente Funktionen bleiben hier auf der CPU,
 * weil GPU-exp/sin/cos andere Bits liefern: exp-Tabellen ueber ts_exp_tab (echte glibc/libmvec wie die Referenz),
 * sincos der Phasen skalar ueber glibc (wie ts_h2c). Die S-Werte je Klasse gehen unveraendert hinauf; sqrt und
 * Multiplikation fuer die Skalierung rechnet die GPU (korrekt gerundet, T0.2/G0 geprueft).
 *
 * Alle Ausgaben sind frequenzverschraenkt: Wert x der Lane l (Frequenz) liegt bei [x * 32 + l].
 * Uebersetzen wie ts_h2c.c: -O2 -ffp-contract=off -fno-fast-math -fno-builtin-sincos -fno-builtin-sin -fno-builtin-cos,
 * OHNE Vektorisierung von sincos (-fno-tree-vectorize), sonst libmvec.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void ts_exp_tab(const int32_t *k_, const double *distk, const double *distuk, const double *uhub_unused, const double *freq_,
                const double *incdec_, const double *incohb_, double *tab);

#define TSGP_L 32

/* exp-Tabellen: tab[(x)*32 + l] fuer x = 0..2K-1, Frequenz der Lane l = freq[l] (nl Lanes, Rest wie die letzte). */
void tsgp_exp(int32_t K, const double *distk, const double *distuk, double uhub, const double *freq, int32_t nl,
              double incdec, double incohb, double *tab, double *tmp /* 2K */)
{
   for (int32_t l = 0; l < TSGP_L; ++l) {
      const double f = freq[l < nl ? l : nl - 1];
      ts_exp_tab(&K, distk, distuk, &uhub, &f, &incdec, &incohb, tmp);
      for (int32_t x = 0; x < 2 * K; ++x) tab[(size_t)x * TSGP_L + l] = tmp[x];
   }
}

/* sincos der Phasen fuer Komponente ivec (1-basiert): cs/sn[j*32 + l] = cos/sin(PhaseAngles(j+1, ifreq[l], ivec)). */
void tsgp_phasen(int32_t n, const double *phase, int32_t ldp, int32_t nfp, const int32_t *ifreq, int32_t nl, int32_t ivec,
                 double *cs, double *sn)
{
   for (int32_t l = 0; l < TSGP_L; ++l) {
      const int32_t f = ifreq[l < nl ? l : nl - 1];
      const double *ph = phase + (size_t)ldp * (((size_t)f - 1) + (size_t)nfp * ((size_t)ivec - 1));
      for (int32_t j = 0; j < n; ++j) {
         double s, c;
         sincos(ph[j], &s, &c);                                /* glibc sincos wie die Referenz (H2Coeffs: COS, SIN) */
         cs[(size_t)j * TSGP_L + l] = c; sn[(size_t)j * TSGP_L + l] = s;
      }
   }
}

/* S-Werte je Klasse: sv[a*32 + l] = S(ifreq[l], rep[a]+1, ivec). */
void tsgp_s(int32_t m, const int32_t *rep, const double *S, int32_t nfs, int32_t nps, const int32_t *ifreq, int32_t nl,
            int32_t ivec, double *sv)
{
   for (int32_t l = 0; l < TSGP_L; ++l) {
      const int32_t f = ifreq[l < nl ? l : nl - 1];
      for (int32_t a = 0; a < m; ++a)
         sv[(size_t)a * TSGP_L + l] = S[((size_t)f - 1) + (size_t)nfs * ((size_t)rep[a] + (size_t)nps * ((size_t)ivec - 1))];
   }
}

/* Klassen der Punkte nach bitgleichen S-Spalten (ueber alle Frequenzen und die Komponenten ivec0..ivec0+ncomp-1):
 * kls[i] = Klasse von Punkt i, rep[a] = erster Punkt der Klasse a. Rueckgabe: Klassenzahl m, oder -1, wenn > maxm.
 * Zwei Punkte derselben Klasse haben fuer jede Frequenz dieselben S-Bits, also dieselben Skalierungsfaktoren. */
int32_t tsgp_klassen(int32_t n, const double *S, int32_t nfs, int32_t nps, int32_t nfreq, int32_t ivec0, int32_t ncomp,
                     int32_t *kls, int32_t *rep, int32_t maxm)
{
   uint64_t *h = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)n);
   if (!h) return -1;
   for (int32_t i = 0; i < n; ++i) {                          /* FNV-1a ueber die Bitmuster der S-Spalte */
      uint64_t x = 1469598103934665603ULL;
      for (int32_t c = 0; c < ncomp; ++c) {
         const double *col = S + (size_t)nfs * ((size_t)i + (size_t)nps * ((size_t)(ivec0 - 1 + c)));
         for (int32_t f = 0; f < nfreq; ++f) { uint64_t u; memcpy(&u, &col[f], 8); x = (x ^ u) * 1099511628211ULL; }
      }
      h[i] = x;
   }
   int32_t m = 0;
   for (int32_t i = 0; i < n; ++i) {
      int32_t a = -1;
      for (int32_t b = 0; b < m && a < 0; ++b) {
         if (h[rep[b]] != h[i]) continue;
         int gleich = 1;                                       /* Hash ist kein Beweis: Bitmuster vollstaendig vergleichen */
         for (int32_t c = 0; c < ncomp && gleich; ++c) {
            const double *x = S + (size_t)nfs * ((size_t)i + (size_t)nps * ((size_t)(ivec0 - 1 + c)));
            const double *y = S + (size_t)nfs * ((size_t)rep[b] + (size_t)nps * ((size_t)(ivec0 - 1 + c)));
            gleich = memcmp(x, y, sizeof(double) * (size_t)nfreq) == 0;
         }
         if (gleich) a = b;
      }
      if (a < 0) { if (m >= maxm) { free(h); return -1; } rep[m] = i; a = m++; }
      kls[i] = a;
   }
   free(h);
   return m;
}

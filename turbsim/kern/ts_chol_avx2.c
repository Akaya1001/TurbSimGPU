/* ts_chol_avx2.c — Nachlauf-Update, Skalierung und Rang-1-Baustein mit AVX2+FMA (aus ts_chol_avx512.c abgeleitet).
 * Uebersetzen mit: -O2 -mavx2 -mfma -ffp-contract=off -fno-fast-math
 *
 * Mikrokachel: 12 Zeilen (3 ymm) x 4 Spalten. Der Akkumulator wird AUS A geladen,
 * die k-Schleife laeuft streng aufsteigend, jede Operation ist genau ein
 * _mm256_fnmadd_pd = fma(-L(c2,k), L(i,k), acc). Spalten mit L(c2,k) == 0 werden
 * ausgelassen (OpenBLAS dspr ueberspringt sie ebenso). Zeilenkacheln, deren
 * Zeilen alle frow-1 >= k1 haben, werden ausgelassen (L(i,k) = 0 fuer alle k < k1;
 * wertgleich, nur Nullvorzeichen koennen abweichen).
 *
 * Zertifizierte No-op-Elimination wie in ts_chol_avx512.c (Kachel 12x4; die Kachelzeilen
 * beginnen auf Vielfachen von 4 und ueberdecken 2-3 Achtergruppen der Skizze).
 */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ts_chol.h"
#include "ts_skip.h"

/* AVX2 hat keine Maskenregister: Masken als __m256i (alle Bits je gueltiger Lane). */
static inline __m256i mk4(int32_t r) { const long long m[4] = { r > 0 ? -1LL : 0, r > 1 ? -1LL : 0, r > 2 ? -1LL : 0, r > 3 ? -1LL : 0 }; return _mm256_loadu_si256((const __m256i *)m); }

#define VL 4
#define TC 4                 /* Spalten je Kachel */
#define TR (3 * VL)          /* Zeilen je Kachel */
#define RB 192               /* Zeilenblock: 192 * 256 * 8 B = 393 KB (L2) */
#define SKIP_MIN_NK 32       /* kurze Panel-Nachlaeufe ohne Skip */
#define VORP_SPIEL 8         /* Vorpruefung mit veraltetem Minimum (nur Aufwandsentscheidung) */

static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }

static void axpy_v(int32_t m, const double *x, double *y, double a)
{
   const __m256d av = _mm256_set1_pd(a);
   int32_t i = 0;
   for (; i + VL <= m; i += VL)
      _mm256_storeu_pd(y + i, _mm256_fnmadd_pd(av, _mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i)));
   if (i < m) {
      const __m256i mk = mk4(m - i);
      _mm256_maskstore_pd(y + i, mk, _mm256_fnmadd_pd(av, _mm256_maskload_pd(x + i, mk), _mm256_maskload_pd(y + i, mk)));
   }
}

static void scal_v(int32_t m, double *x, double r)
{
   const __m256d rv = _mm256_set1_pd(r);
   int32_t i = 0;
   for (; i + VL <= m; i += VL) _mm256_storeu_pd(x + i, _mm256_mul_pd(_mm256_loadu_pd(x + i), rv));
   if (i < m) {
      const __m256i mk = mk4(m - i);
      _mm256_maskstore_pd(x + i, mk, _mm256_mul_pd(_mm256_maskload_pd(x + i, mk), rv));
   }
}

/* Kleinstes ts_es der gueltigen Akkumulatoren (Betragsminimum; ungueltige Lanes -> +Inf; NaN -> nie auslassen). */
static inline int32_t es_min_kachel(__m256d acc[TC][3], int32_t nc, __m256i m0, __m256i m1, __m256i m2)
{
   const __m256d inf = _mm256_set1_pd(INFINITY), absm = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
   __m256d mn = inf;
   int nan = 0;
   for (int jj = 0; jj < nc; ++jj) {
      const __m256d a0 = _mm256_blendv_pd(inf, _mm256_and_pd(acc[jj][0], absm), _mm256_castsi256_pd(m0));
      const __m256d a1 = _mm256_blendv_pd(inf, _mm256_and_pd(acc[jj][1], absm), _mm256_castsi256_pd(m1));
      const __m256d a2 = _mm256_blendv_pd(inf, _mm256_and_pd(acc[jj][2], absm), _mm256_castsi256_pd(m2));
      nan |= _mm256_movemask_pd(_mm256_cmp_pd(a0, a0, _CMP_UNORD_Q)) | _mm256_movemask_pd(_mm256_cmp_pd(a1, a1, _CMP_UNORD_Q))
           | _mm256_movemask_pd(_mm256_cmp_pd(a2, a2, _CMP_UNORD_Q));
      mn = _mm256_min_pd(mn, _mm256_min_pd(a0, _mm256_min_pd(a1, a2)));
   }
   if (nan) return -2 * TS_E_NAN;
   double v[4]; _mm256_storeu_pd(v, mn);
   double m = v[0]; for (int t = 1; t < 4; ++t) if (v[t] < m) m = v[t];
   return ts_es(m);
}

/* Eine Mikrokachel: Zeilen [i, i+rows) (rows <= TR), Spalten colp[0..TC-1] (ungueltige
 * Spalten zeigen auf scratch und haben in Cv nur Nullen). P: gepackte Panelzeilen
 * (k-major, rb je k), Cv: Spaltenwerte L(c2+jj,k) k-major (TC je k).
 * s8: Gruppen-Skizze ab dem ersten Chunk des k-Bereichs oder NULL; gc = c2/8; nc gueltige Spalten. */
static inline void kachel(int32_t rows, double *const colp[TC], int32_t i, int32_t ioff, const double *P, int32_t rb,
                          const double *Cv, int32_t nk, const int16_t *s8, int32_t ng, int32_t gc, int32_t nc)
{
   const __m256i m0 = mk4(rows), m1 = mk4(rows - 4), m2 = mk4(rows - 8);
   __m256d acc[TC][3];
#pragma GCC unroll 4
   for (int jj = 0; jj < TC; ++jj) {
      acc[jj][0] = _mm256_maskload_pd(colp[jj] + i, m0);
      acc[jj][1] = _mm256_maskload_pd(colp[jj] + i + 4, m1);
      acc[jj][2] = _mm256_maskload_pd(colp[jj] + i + 8, m2);
   }
   const int32_t ga = i / 8, gb = (i + rows - 1) / 8;            /* Achtergruppen der Kachelzeilen */
   int32_t es = 0, frisch = 0;
   if (s8) { es = es_min_kachel(acc, nc, m0, m1, m2); frisch = 1; }
   for (int32_t q = 0; q < nk; q += TS_SK_KC) {
      const int32_t qe = imin(q + TS_SK_KC, nk);
      if (s8 && qe - q == TS_SK_KC) {
         const int16_t *z = s8 + (size_t)(q / TS_SK_KC) * (size_t)ng;
         int32_t rm = z[ga];
         for (int32_t g = ga + 1; g <= gb; ++g) if (z[g] > rm) rm = z[g];
         const int32_t schranke = rm + z[gc];
         if (schranke <= es - TS_SKIP_SCHWELLE + VORP_SPIEL) {
            if (!frisch) { es = es_min_kachel(acc, nc, m0, m1, m2); frisch = 1; }
            if (schranke <= es - TS_SKIP_SCHWELLE) continue;     /* alle 8 Schritte sind No-ops */
         }
      }
      for (int32_t kk = q; kk < qe; ++kk) {
         const double *pk = P + (size_t)kk * rb + ioff;
         const __m256d l0 = _mm256_loadu_pd(pk), l1 = _mm256_loadu_pd(pk + 4), l2 = _mm256_loadu_pd(pk + 8);
         const double *cv = Cv + (size_t)kk * TC;
#pragma GCC unroll 4
         for (int jj = 0; jj < TC; ++jj) {
            const double lck = cv[jj];
            if (lck != 0.0) {
               const __m256d b = _mm256_set1_pd(lck);
               acc[jj][0] = _mm256_fnmadd_pd(b, l0, acc[jj][0]);
               acc[jj][1] = _mm256_fnmadd_pd(b, l1, acc[jj][1]);
               acc[jj][2] = _mm256_fnmadd_pd(b, l2, acc[jj][2]);
            }
         }
      }
      frisch = 0;
   }
#pragma GCC unroll 4
   for (int jj = 0; jj < TC; ++jj) {
      _mm256_maskstore_pd(colp[jj] + i, m0, acc[jj][0]);
      _mm256_maskstore_pd(colp[jj] + i + 4, m1, acc[jj][1]);
      _mm256_maskstore_pd(colp[jj] + i + 8, m2, acc[jj][2]);
   }
}

static void trail_v(int32_t n, double *ap, int32_t k0, int32_t k1, int32_t from, int32_t to, const int32_t *frow,
                    const ts_skizze *skz)
{
   if (from >= to || k0 >= k1) return;
   const int32_t nk = k1 - k0;
   const int16_t *s8 = (skz && nk >= SKIP_MIN_NK && k0 % TS_SK_KC == 0) ? skz->sk8 + (size_t)(k0 / TS_SK_KC) * (size_t)skz->ng : 0;
   const int32_t ng = skz ? skz->ng : 0;
   const size_t prb = (size_t)RB + 2 * TR;                 /* gepolsterte Zeilen je k (Lesen hinter rb erlaubt) */
   double *P  = (double *)aligned_alloc(64, sizeof(double) * prb * (size_t)nk);
   double *Cv = (double *)aligned_alloc(64, sizeof(double) * TC * (size_t)nk);
   double scratch[TR + 8];
   if (!P || !Cv) { free(P); free(Cv); ts_chol_isa_scalar.trail(n, ap, k0, k1, from, to, frow, skz); return; }
   memset(scratch, 0, sizeof scratch);

   for (int32_t r0 = from; r0 < n; r0 += RB) {
      const int32_t r1 = imin(n, r0 + RB), rb = r1 - r0;
      const int32_t cend = imin(to, r1);                      /* Spalten mit Zeilen in diesem Block */
      if (from >= cend) break;
      /* Panelzeilen packen: P[kk*prb + (i-r0)] = L(i, k0+kk), Polster mit Nullen */
      for (int32_t kk = 0; kk < nk; ++kk) {
         const double *lk = ap + ts_col0(n, k0 + kk) - (k0 + kk);
         memcpy(P + (size_t)kk * prb, lk + r0, sizeof(double) * (size_t)rb);
         memset(P + (size_t)kk * prb + rb, 0, sizeof(double) * (prb - (size_t)rb));
      }
      for (int32_t c2 = from; c2 < cend; c2 += TC) {
         const int32_t nc = imin(TC, cend - c2);
         double *colp[TC];
         for (int jj = 0; jj < TC; ++jj)
            colp[jj] = (jj < nc) ? ap + ts_col0(n, c2 + jj) - (c2 + jj) : scratch - 0;   /* scratch: Lesen/Schreiben maskiert */
         /* Spaltenwerte L(c2+jj, k) k-major, ungueltige Spalten = 0 -> werden uebersprungen */
         for (int32_t kk = 0; kk < nk; ++kk) {
            const double *lk = ap + ts_col0(n, k0 + kk) - (k0 + kk);
            for (int jj = 0; jj < TC; ++jj) Cv[(size_t)kk * TC + jj] = (jj < nc) ? lk[c2 + jj] : 0.0;
         }
         /* Dreieckskopf: Zeilen [max(r0,c2), min(r1, c2+TC)) skalar, nur j <= i */
         const int32_t ih0 = imax(r0, c2), ih1 = imin(r1, c2 + TC);
         for (int32_t i = ih0; i < ih1; ++i) {
            if (frow && frow[i] - 1 >= k1) continue;
            for (int jj = 0; jj < nc && c2 + jj <= i; ++jj) {
               double s = colp[jj][i];
               for (int32_t kk = 0; kk < nk; ++kk) {
                  const double lck = Cv[(size_t)kk * TC + jj];
                  if (lck != 0.0) s = fma(-lck, P[(size_t)kk * prb + (i - r0)], s);
               }
               colp[jj][i] = s;
            }
         }
         /* Vektorteil: Zeilen ab max(r0, c2+TC) */
         for (int32_t i = imax(r0, c2 + TC); i < r1; i += TR) {
            const int32_t rows = imin(TR, r1 - i);
            if (frow) {
               int32_t fmin = frow[i];
               for (int32_t t = 1; t < rows; ++t) if (frow[i + t] < fmin) fmin = frow[i + t];
               if (fmin - 1 >= k1) continue;                /* ganze Kachel: L(i,k) = 0 fuer alle k < k1 */
            }
            /* ungueltige Spalten muessen beim Laden/Speichern unschaedlich sein: scratch-Zeiger so legen,
               dass colp[jj] + i auf scratch zeigt */
            for (int jj = nc; jj < TC; ++jj) colp[jj] = scratch - i;
            kachel(rows, colp, i, i - r0, P, (int32_t)prb, Cv, nk, s8, ng, c2 / 8, nc);
         }
      }
   }
   free(P); free(Cv);
}

/* Skizze: die skalare Fassung genuegt hier (AVX2 ist nur der Rueckfall ohne AVX-512). */
static void sketch_v(int32_t n, const double *ap, int32_t q, const ts_skizze *skz)
{
   ts_chol_isa_scalar.sketch(n, ap, q, skz);
}

const ts_chol_isa ts_chol_isa_avx2 = { axpy_v, scal_v, trail_v, sketch_v, "avx2" };

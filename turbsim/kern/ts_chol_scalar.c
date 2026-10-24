/* Skalare Referenzfassung der dpptrf-Kette (Basis-x86-64, kein FMA-Befehl noetig:
 * fma() ist in glibc ein IFUNC auf die Hardware-FMA). Langsam, aber die Messlatte. */
#include <math.h>
#include "ts_chol.h"
#include "ts_skip.h"

static void axpy_s(int32_t m, const double *x, double *y, double a)
{
   for (int32_t i = 0; i < m; ++i) y[i] = fma(-a, x[i], y[i]);
}
static void scal_s(int32_t m, double *x, double r)
{
   for (int32_t i = 0; i < m; ++i) x[i] = x[i] * r;
}
/* Nachlauf: je Spalte c2, je Zeile i: Kette ueber k = k0..k1-1 aufsteigend. Mit Skizze wird je
 * Element ein ganzer Chunk ausgelassen, wenn er zertifiziert No-ops enthaelt (ts_skip.h). */
static void trail_s(int32_t n, double *ap, int32_t k0, int32_t k1, int32_t from, int32_t to, const int32_t *frow,
                    const ts_skizze *skz)
{
   if (skz && k0 % TS_SK_KC) skz = 0;                   /* Skizze gilt nur fuer Chunks ab Vielfachen von 8 */
   for (int32_t c2 = from; c2 < to; ++c2) {
      double *col = ap + ts_col0(n, c2) - c2;          /* col[i] = A(i,c2) */
      for (int32_t i = c2; i < n; ++i) {
         if (frow && frow[i] - 1 >= k1) continue;      /* L(i,k) = 0 fuer alle k < k1 */
         double s = col[i];
         for (int32_t q = k0; q < k1; q += TS_SK_KC) {
            const int32_t qe = (q + TS_SK_KC < k1) ? q + TS_SK_KC : k1;
            if (skz && qe - q == TS_SK_KC) {
               const int16_t *sk = skz->sk + (size_t)(q / TS_SK_KC) * (size_t)n;
               if (sk[i] + sk[c2] <= ts_es(s) - TS_SKIP_SCHWELLE) continue;
            }
            for (int32_t k = q; k < qe; ++k) {
               const double *lk = ap + ts_col0(n, k) - k;  /* lk[j] = L(j,k) */
               const double lck = lk[c2];
               if (lck != 0.0) s = fma(-lck, lk[i], s);
            }
         }
         col[i] = s;
      }
   }
}
/* Skizze des Chunks ab Spalte q (Spalten q..q+7 fertig) fuer die Zeilen ab q+8. */
static void sketch_s(int32_t n, const double *ap, int32_t q, const ts_skizze *skz)
{
   const int32_t K = q / TS_SK_KC, i0 = q + TS_SK_KC;
   int16_t *sk = skz->sk + (size_t)K * (size_t)n;
   for (int32_t i = i0; i < n; ++i) {
      int32_t m = TS_E_NULL;
      for (int32_t k = q; k < i0; ++k) {
         const int32_t e = ts_e(ap[ts_col0(n, k) + (size_t)(i - k)]);
         if (e > m) m = e;
      }
      sk[i] = (int16_t)m;
   }
   int16_t *s8 = skz->sk8 + (size_t)K * (size_t)skz->ng;
   for (int32_t g = i0 / 8; g < skz->ng; ++g) {
      int32_t m = TS_E_NULL;
      for (int32_t i = 8 * g; i < 8 * g + 8 && i < n; ++i) if (sk[i] > m) m = sk[i];
      s8[g] = (int16_t)m;
   }
}
const ts_chol_isa ts_chol_isa_scalar = { axpy_s, scal_s, trail_s, sketch_s, "scalar" };

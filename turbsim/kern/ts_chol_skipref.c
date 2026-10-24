/* ts_chol_skipref.c — skalare Referenz der dpptrf-Kette MIT zertifizierter No-op-Elimination
 * (Plan 14.10.2026, Stufe A1). Zweck: Messlatte und Statistik fuer die Skip-Regel aus ts_skip.h,
 * nicht Tempo. Nicht Teil des TurbSim-Binarys; die Tests und skip_pruefung.py binden es ein.
 *
 * Ablauf links-blickend je Spalte c: s[i] = A(i,c) fuer i >= c; k = 0..c-1 AUFSTEIGEND aussen,
 * Zeilen i innen -> je Element bleibt die Kette der Spezifikation (ts_chol.h) erhalten:
 *   wenn L(c,k) != 0: s = fma(-L(c,k), L(i,k), s); dann L(c,c) = sqrt(s), r = 1.0/L(c,c), L(i,c) = s*r.
 *
 * Modi (skip):
 *   0  reine Spezifikation — muss bitgleich zu ts_chol_spec ohne Zeilenprofil sein (inkl. Nullvorzeichen)
 *   1  Elementregel ts_skip je k
 *   2  zweistufig: Sketch je k-Chunk der Breite kc; nur in offenen Chunks die Elementregel je k
 * pruefen != 0: jeden uebersprungenen Schritt trotzdem rechnen und mit s vergleichen.
 *
 * Zaehler z[0..9] (int64, werden addiert):
 *   0  Spezifikationsschritte (L(c,k) != 0)          1  ausgefuehrte Schritte
 *   2  Schritte in offenen Chunks (Sketch-Granularitaet; skip 2, sonst = z[0])
 *   3  Verletzungen: uebersprungen, aber fma(-a,b,s) != s und nicht beide Null (muss 0 sein)
 *   4  geprufte uebersprungene Schritte               5  davon nur Nullvorzeichen anders
 *   6  freie Elemente (i>c, mind. ein Spezifikationsschritt, keiner ausgefuehrt)
 *   7  Elemente i>c mit mind. einem Spezifikationsschritt
 *   8  Invariante: per Sketch ausgelassene Schritte, die die Elementregel NICHT erfuellen (muss 0 sein)
 *   9  reserviert
 * Pruefschwelle: ts_skipref_schwelle_setzen() aendert die Schwelle NUR in dieser Referenz (Positivkontrolle
 * der Pruefinstrumente: mit 55 muessen Verletzungen gezaehlt werden). Vorgabe TS_SKIP_SCHWELLE.
 * info wie LAPACK: 0 ok, sonst Ordnung des ersten nicht positiv definiten Minors.
 *
 * Uebersetzen mit -ffp-contract=off -fno-fast-math (nur das explizite fma() ist eine FMA).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "ts_chol.h"
#include "ts_skip.h"

#define SR_BLOCK 256          /* Zeilenblock je Faden */

static int32_t sr_schwelle = TS_SKIP_SCHWELLE;
void ts_skipref_schwelle_setzen(int32_t s) { sr_schwelle = s; }
/* Regel mit der Pruefschwelle; bei der Vorgabe identisch zu ts_skip (ts_skip.h). */
static inline int sr_skip(double a, double b, double s) { return ts_e(a) + ts_e(b) <= ts_es(s) - sr_schwelle; }

static inline int bits_gleich(double x, double y)
{
   uint64_t a, b; memcpy(&a, &x, sizeof a); memcpy(&b, &y, sizeof b);
   return a == b;
}

void ts_chol_skipref(const int32_t *n_, double *ap, const int32_t *skip_, const int32_t *kc_,
                     const int32_t *pruefen_, int64_t *z, int32_t *info)
{
   const int32_t n = *n_, skip = *skip_, pruefen = *pruefen_;
   const int32_t kc = (*kc_ > 0) ? *kc_ : 8;
   const int32_t nk = (n + kc - 1) / kc;
   int16_t *sk = 0;
   if (skip == 2) {
      sk = (int16_t *)malloc((size_t)n * (size_t)nk * sizeof(int16_t));
      if (!sk) { *info = -1; return; }
      for (size_t e = 0; e < (size_t)n * (size_t)nk; ++e) sk[e] = TS_E_OFFEN;
   }
   int64_t z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0, z6 = 0, z7 = 0, z8 = 0;
   int32_t inf = 0;

   for (int32_t c = 0; c < n; ++c) {
      double *s = ap + ts_col0(n, c) - c;              /* s[i] = A(i,c), i >= c (in place) */
      const int64_t arbeit = (int64_t)(n - c) * (int64_t)c;
      #pragma omp parallel for schedule(dynamic, 1) if (arbeit > 100000) \
              reduction(+ : z0, z1, z2, z3, z4, z5, z6, z7, z8)
      for (int32_t ib = c; ib < n; ib += SR_BLOCK) {
         const int32_t ie = (ib + SR_BLOCK < n) ? ib + SR_BLOCK : n;
         unsigned char offen[SR_BLOCK], hat_spec[SR_BLOCK], hat_exec[SR_BLOCK];
         memset(hat_spec, 0, sizeof hat_spec); memset(hat_exec, 0, sizeof hat_exec);
         for (int32_t k0 = 0; k0 < c; k0 += kc) {
            const int32_t k1 = (k0 + kc < c) ? k0 + kc : c;
            const int32_t K = k0 / kc;
            /* Sketch nur fuer vollstaendige Chunks (alle Spalten fertig skaliert) */
            if (skip == 2 && k0 + kc <= c) {
               const int32_t sc = sk[(size_t)c * nk + K];
               for (int32_t i = ib; i < ie; ++i)
                  offen[i - ib] = !(sk[(size_t)i * nk + K] + sc <= ts_es(s[i]) - sr_schwelle);
            } else {
               memset(offen, 1, sizeof offen);
            }
            for (int32_t k = k0; k < k1; ++k) {
               const double *lk = ap + ts_col0(n, k) - k;  /* lk[j] = L(j,k) */
               const double lck = lk[c];
               if (lck == 0.0) continue;                  /* Referenz-Skip der Spezifikation */
               for (int32_t i = ib; i < ie; ++i) {
                  const double lik = lk[i];
                  ++z0; hat_spec[i - ib] = 1;
                  int zert;
                  if (!offen[i - ib]) { zert = 1; if (!sr_skip(lck, lik, s[i])) ++z8; }
                  else { ++z2; zert = (skip >= 1) && sr_skip(lck, lik, s[i]); }
                  if (skip == 0) zert = 0;
                  if (zert) {
                     if (pruefen) {
                        const double t = fma(-lck, lik, s[i]);
                        ++z4;
                        if (!bits_gleich(t, s[i])) {
                           if (t == 0.0 && s[i] == 0.0) ++z5; else ++z3;
                        }
                     }
                  } else {
                     s[i] = fma(-lck, lik, s[i]);
                     ++z1; hat_exec[i - ib] = 1;
                  }
               }
            }
         }
         for (int32_t i = ib; i < ie; ++i) {
            if (i == c || !hat_spec[i - ib]) continue;
            ++z7; if (!hat_exec[i - ib]) ++z6;
         }
      }
      /* Diagonale wie faktor_ungeblockt (ts_kern.c): NaN laeuft durch */
      const double d = s[c];
      if (d <= 0.0) { inf = c + 1; break; }
      const double L = sqrt(d);
      s[c] = L;
      const double r = 1.0 / L;
      for (int32_t i = c + 1; i < n; ++i) s[i] = s[i] * r;
      /* Sketch eines fertigen Chunks: Zeilen unterhalb des Chunks (nur die werden je gelesen) */
      if (skip == 2 && (c + 1) % kc == 0) {
         const int32_t K = c / kc, k0 = K * kc;
         #pragma omp parallel for schedule(static) if (n - c > 4096)
         for (int32_t i = c + 1; i < n; ++i) {
            int32_t m = TS_E_NULL;
            for (int32_t k = k0; k <= c; ++k) {
               const int32_t e = ts_e(ap[ts_col0(n, k) + (size_t)(i - k)]);
               if (e > m) m = e;
            }
            sk[(size_t)i * nk + K] = (int16_t)m;
         }
      }
   }
   free(sk);
   z[0] += z0; z[1] += z1; z[2] += z2; z[3] += z3; z[4] += z4; z[5] += z5; z[6] += z6; z[7] += z7; z[8] += z8;
   *info = inf;
}

/* Pruefhilfe fuer die Regel selbst: m Tripel (a,b,s) gegen die echte fma (mit der Pruefschwelle).
 *   z[0] zertifiziert   z[1] Verletzungen   z[2] zertifiziert, nur Nullvorzeichen anders
 *   z[3] echte No-ops (fma == s bitweise oder beide Null)   z[4] davon nicht zertifiziert
 *   z[5] Tripel, bei denen ts_skip (Produktion) und die Pruefregel verschieden entscheiden */
void ts_skip_pruef_tripel(const int64_t *m_, const double *a, const double *b, const double *s, int64_t *z)
{
   const int64_t m = *m_;
   int64_t z0 = 0, z1 = 0, z2 = 0, z3 = 0, z4 = 0, z5 = 0;
   #pragma omp parallel for schedule(static) reduction(+ : z0, z1, z2, z3, z4, z5)
   for (int64_t j = 0; j < m; ++j) {
      const double t = fma(-a[j], b[j], s[j]);
      const int gleich = bits_gleich(t, s[j]);
      const int null2 = (t == 0.0 && s[j] == 0.0);
      const int zert = sr_skip(a[j], b[j], s[j]);
      if (zert != ts_skip(a[j], b[j], s[j])) ++z5;
      if (gleich || null2) { ++z3; if (!zert) ++z4; }
      if (zert) { ++z0; if (!gleich) { if (null2) ++z2; else ++z1; } }
   }
   z[0] += z0; z[1] += z1; z[2] += z2; z[3] += z3; z[4] += z4; z[5] += z5;
}

/* ctypes-Zugang zu den Inline-Funktionen */
int32_t ts_skip_e(double x) { return ts_e(x); }
int32_t ts_skip_es(double s) { return ts_es(s); }
int32_t ts_skip_regel(double a, double b, double s) { return ts_skip(a, b, s); }

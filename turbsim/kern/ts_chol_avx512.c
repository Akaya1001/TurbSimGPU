/* ts_chol_avx512.c — Nachlauf-Update, Skalierung und Rang-1-Baustein mit AVX-512.
 * Uebersetzen mit: -O2 -mavx512f -mavx512dq -mavx512vl -mfma -ffp-contract=off -fno-fast-math
 *
 * Mikrokachel: 24 Zeilen (3 zmm) x 8 Spalten. Der Akkumulator wird AUS A geladen,
 * die k-Schleife laeuft streng aufsteigend, jede Operation ist genau ein
 * _mm512_fnmadd_pd = fma(-L(c2,k), L(i,k), acc). Spalten mit L(c2,k) == 0 werden
 * ausgelassen (OpenBLAS dspr ueberspringt sie ebenso). Zeilenkacheln, deren
 * Zeilen alle frow-1 >= k1 haben, werden ausgelassen (L(i,k) = 0 fuer alle k < k1;
 * wertgleich, nur Nullvorzeichen koennen abweichen).
 *
 * Zertifizierte No-op-Elimination (ts_skip.h, Plan 14.10.2026 Stufe A2): je Kachel und k-Chunk
 * (TS_SK_KC = 8 Spalten) wird der ganze Chunk ausgelassen, wenn
 *   max_{Kachelzeilen} sk8 + max_{Kachelspalten} sk8 <= min_{Kachel} ts_es(acc) - TS_SKIP_SCHWELLE.
 * Dann liefert jede der FMAs bitweise acc (bzw. eine Null mit anderem Vorzeichen). Das Minimum
 * ueber die Akkumulatoren ist dabei immer frisch: neu bestimmt nach jedem ausgefuehrten Chunk,
 * bevor ein spaeterer Chunk ausgelassen werden darf. Ausfuehren ist immer exakt — die Vorpruefung
 * mit dem alten Minimum (VORP_SPIEL) entscheidet nur, ob sich das Neubestimmen lohnt.
 *
 * Feste Kosten: Eine Kachel, deren saemtliche Chunks schon mit dem Schatten esh (untere Schranke
 * von ts_es je 8x8-Block, nach jeder bearbeiteten Kachel neu geschrieben) zertifiziert sind, wird
 * gar nicht geladen; Panelzeilen und Spaltenwerte werden erst gepackt, wenn eine Kachel sie braucht.
 * Der skalare Dreieckskopf prueft je Element und Chunk die Element-Skizze sk.
 */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ts_chol.h"
#include "ts_skip.h"

#define VL 8
#define TC 8                 /* Spalten je Kachel */
#define TR (3 * VL)          /* Zeilen je Kachel */
#define RB 192               /* Zeilenblock: 192 * 256 * 8 B = 393 KB (L2) */
#define SKIP_MIN_NK 32       /* kurze Panel-Nachlaeufe (k-Bereich < 32) ohne Skip: Pruefaufwand > Ertrag */
#define VORP_SPIEL 0         /* Vorpruefung mit veraltetem Minimum (Binaden Spielraum; gemessen unkritisch) */

static inline int32_t imin(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }

static void axpy_v(int32_t m, const double *x, double *y, double a)
{
   const __m512d av = _mm512_set1_pd(a);
   int32_t i = 0;
   for (; i + VL <= m; i += VL)
      _mm512_storeu_pd(y + i, _mm512_fnmadd_pd(av, _mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i)));
   if (i < m) {
      const __mmask8 mk = (__mmask8)((1u << (m - i)) - 1u);
      _mm512_mask_storeu_pd(y + i, mk, _mm512_fnmadd_pd(av, _mm512_maskz_loadu_pd(mk, x + i), _mm512_maskz_loadu_pd(mk, y + i)));
   }
}

static void scal_v(int32_t m, double *x, double r)
{
   const __m512d rv = _mm512_set1_pd(r);
   int32_t i = 0;
   for (; i + VL <= m; i += VL) _mm512_storeu_pd(x + i, _mm512_mul_pd(_mm512_loadu_pd(x + i), rv));
   if (i < m) {
      const __mmask8 mk = (__mmask8)((1u << (m - i)) - 1u);
      _mm512_mask_storeu_pd(x + i, mk, _mm512_mul_pd(_mm512_maskz_loadu_pd(mk, x + i), rv));
   }
}

/* ts_e fuer 8 Doubles (Bitfeld wie ts_skip.h): Null -> TS_E_NULL, subnormal -> -1023, Inf/NaN -> TS_E_NAN. */
static inline __m512i e_vec(__m512d x)
{
   const __m512i u = _mm512_castpd_si512(x);
   const __m512i f = _mm512_and_si512(_mm512_srli_epi64(u, 52), _mm512_set1_epi64(0x7FF));
   __m512i e = _mm512_sub_epi64(f, _mm512_set1_epi64(1023));
   const __mmask8 f0 = _mm512_cmpeq_epi64_mask(f, _mm512_setzero_si512());
   const __mmask8 null = _mm512_mask_cmpeq_epi64_mask(f0, _mm512_slli_epi64(u, 1), _mm512_setzero_si512());
   e = _mm512_mask_mov_epi64(e, f0, _mm512_set1_epi64(-1023));
   e = _mm512_mask_mov_epi64(e, null, _mm512_set1_epi64(TS_E_NULL));
   e = _mm512_mask_mov_epi64(e, _mm512_cmpeq_epi64_mask(f, _mm512_set1_epi64(0x7FF)), _mm512_set1_epi64(TS_E_NAN));
   return e;
}

/* Kleinstes ts_es der gueltigen Akkumulatoren einer Kachel (untere Schranke fuer alle Elemente).
 * ts_es ist monoton in |s|, also genuegt das Minimum der Betraege; NaN irgendwo -> nie auslassen. */
static inline int32_t es_min_kachel(__m512d acc[TC][3], int32_t nc, __mmask8 m0, __mmask8 m1, __mmask8 m2)
{
   const __m512d inf = _mm512_set1_pd(INFINITY);
   __m512d mn = inf;
   __mmask8 nan = 0;
   for (int jj = 0; jj < nc; ++jj) {
      const __m512d a0 = _mm512_abs_pd(acc[jj][0]), a1 = _mm512_abs_pd(acc[jj][1]), a2 = _mm512_abs_pd(acc[jj][2]);
      nan |= _mm512_mask_cmp_pd_mask(m0, a0, a0, _CMP_UNORD_Q) | _mm512_mask_cmp_pd_mask(m1, a1, a1, _CMP_UNORD_Q)
           | _mm512_mask_cmp_pd_mask(m2, a2, a2, _CMP_UNORD_Q);
      mn = _mm512_mask_min_pd(mn, m0, mn, a0);
      mn = _mm512_mask_min_pd(mn, m1, mn, a1);
      mn = _mm512_mask_min_pd(mn, m2, mn, a2);
   }
   if (nan) return -2 * TS_E_NAN;
   return ts_es(_mm512_reduce_min_pd(mn));
}

/* Gepackte Panelzeilen P (je Zeilenblock) und Spaltenwerte Cv (je Spaltenkachel), chunkweise erst bei
 * Bedarf gefuellt: bei hohen Frequenzen braucht kaum eine Kachel mehr als ein paar Chunks. */
typedef struct {
   double *P; size_t prb; unsigned char *pq;     /* pq[K]: Chunk K der Panelzeilen gepackt */
   double *Cv; unsigned char *cq;                /* cq[K]: Chunk K der Spaltenwerte da */
   const double *ap; int32_t n, k0, r0, rb, c2, nc;
} paket;

static inline void chunk_bereit(paket *pk, int32_t q, int32_t qe)
{
   const int32_t K = q / TS_SK_KC;
   if (!pk->pq[K]) {                                   /* P[kk*prb + (i-r0)] = L(i, k0+kk), Polster mit Nullen */
      for (int32_t kk = q; kk < qe; ++kk) {
         const double *lk = pk->ap + ts_col0(pk->n, pk->k0 + kk) - (pk->k0 + kk);
         memcpy(pk->P + (size_t)kk * pk->prb, lk + pk->r0, sizeof(double) * (size_t)pk->rb);
         memset(pk->P + (size_t)kk * pk->prb + pk->rb, 0, sizeof(double) * (pk->prb - (size_t)pk->rb));
      }
      pk->pq[K] = 1;
   }
   if (!pk->cq[K]) {                                   /* Cv[kk*TC + jj] = L(c2+jj, k0+kk), ungueltige Spalten = 0 */
      for (int32_t kk = q; kk < qe; ++kk) {
         const double *lk = pk->ap + ts_col0(pk->n, pk->k0 + kk) - (pk->k0 + kk);
         for (int jj = 0; jj < TC; ++jj) pk->Cv[(size_t)kk * TC + jj] = (jj < pk->nc) ? lk[pk->c2 + jj] : 0.0;
      }
      pk->cq[K] = 1;
   }
}

/* ts_es-Minimum einer Achtergruppe der Kachel (Vektor v = 0/1/2) fuer den Schatten. */
static inline int32_t es_min_gruppe(__m512d acc[TC][3], int v, int32_t nc, __mmask8 m)
{
   const __m512d inf = _mm512_set1_pd(INFINITY);
   __m512d mn = inf;
   __mmask8 nan = 0;
   for (int jj = 0; jj < nc; ++jj) {
      const __m512d a = _mm512_abs_pd(acc[jj][v]);
      nan |= _mm512_mask_cmp_pd_mask(m, a, a, _CMP_UNORD_Q);
      mn = _mm512_mask_min_pd(mn, m, mn, a);
   }
   if (nan) return TS_ESH_UNBEKANNT;
   return ts_es(_mm512_reduce_min_pd(mn));
}

/* Eine Mikrokachel: Zeilen [i, i+rows) (rows <= TR), Spalten colp[0..TC-1] (ungueltige
 * Spalten zeigen auf scratch und haben in Cv nur Nullen). P: gepackte Panelzeilen
 * (k-major, rb je k), Cv: Spaltenwerte L(c2+jj,k) k-major (TC je k).
 * s8: Gruppen-Skizze ab dem ersten Chunk des k-Bereichs (Zeile je Chunk, Abstand ng) oder NULL;
 * g0 = i/8 (Kachelzeilen), gc = c2/8 (Kachelspalten); nc gueltige Spalten.
 * esh: Schatten der Spaltengruppe (esh[g] fuer Zeilengruppe g) oder NULL; wird am Ende neu geschrieben. */
static inline void kachel(int32_t rows, double *const colp[TC], int32_t i, int32_t ioff, paket *pk,
                          int32_t nk, const int16_t *s8, int32_t ng, int32_t g0, int32_t gc, int32_t nc,
                          int16_t *esh)
{
   const double *P = pk->P, *Cv = pk->Cv;
   const int32_t rb = (int32_t)pk->prb;
   const __mmask8 m0 = (__mmask8)(rows >= 8 ? 0xFF : ((1u << rows) - 1u));
   const __mmask8 m1 = (__mmask8)(rows >= 16 ? 0xFF : (rows > 8 ? ((1u << (rows - 8)) - 1u) : 0));
   const __mmask8 m2 = (__mmask8)(rows >= 24 ? 0xFF : (rows > 16 ? ((1u << (rows - 16)) - 1u) : 0));
   __m512d acc[TC][3];
#pragma GCC unroll 8
   for (int jj = 0; jj < TC; ++jj) {
      acc[jj][0] = _mm512_maskz_loadu_pd(m0, colp[jj] + i);
      acc[jj][1] = _mm512_maskz_loadu_pd(m1, colp[jj] + i + 8);
      acc[jj][2] = _mm512_maskz_loadu_pd(m2, colp[jj] + i + 16);
   }
   int32_t es = 0, frisch = 0;
   if (s8) { es = es_min_kachel(acc, nc, m0, m1, m2); frisch = 1; }
   for (int32_t q = 0; q < nk; q += TS_SK_KC) {
      const int32_t qe = imin(q + TS_SK_KC, nk);
      if (s8 && qe - q == TS_SK_KC) {
         const int16_t *z = s8 + (size_t)(q / TS_SK_KC) * (size_t)ng;
         int32_t rm = z[g0];
         if (rows > 8 && z[g0 + 1] > rm) rm = z[g0 + 1];
         if (rows > 16 && z[g0 + 2] > rm) rm = z[g0 + 2];
         const int32_t schranke = rm + z[gc];
         if (schranke <= es - TS_SKIP_SCHWELLE + VORP_SPIEL) {
            if (!frisch) { es = es_min_kachel(acc, nc, m0, m1, m2); frisch = 1; }
            if (schranke <= es - TS_SKIP_SCHWELLE) continue;     /* alle 8 Schritte sind No-ops */
         }
      }
      chunk_bereit(pk, q, qe);
      for (int32_t kk = q; kk < qe; ++kk) {
         const double *pz = P + (size_t)kk * rb + ioff;
         const __m512d l0 = _mm512_loadu_pd(pz), l1 = _mm512_loadu_pd(pz + 8), l2 = _mm512_loadu_pd(pz + 16);
         const double *cv = Cv + (size_t)kk * TC;
#pragma GCC unroll 8
         for (int jj = 0; jj < TC; ++jj) {
            const double lck = cv[jj];
            if (lck != 0.0) {
               const __m512d b = _mm512_set1_pd(lck);
               acc[jj][0] = _mm512_fnmadd_pd(b, l0, acc[jj][0]);
               acc[jj][1] = _mm512_fnmadd_pd(b, l1, acc[jj][1]);
               acc[jj][2] = _mm512_fnmadd_pd(b, l2, acc[jj][2]);
            }
         }
      }
      frisch = 0;
   }
   if (esh) {
      esh[g0] = (int16_t)es_min_gruppe(acc, 0, nc, m0);
      if (rows > 8) esh[g0 + 1] = (int16_t)es_min_gruppe(acc, 1, nc, m1);
      if (rows > 16) esh[g0 + 2] = (int16_t)es_min_gruppe(acc, 2, nc, m2);
   }
#pragma GCC unroll 8
   for (int jj = 0; jj < TC; ++jj) {
      _mm512_mask_storeu_pd(colp[jj] + i, m0, acc[jj][0]);
      _mm512_mask_storeu_pd(colp[jj] + i + 8, m1, acc[jj][1]);
      _mm512_mask_storeu_pd(colp[jj] + i + 16, m2, acc[jj][2]);
   }
}

/* Alle Chunks einer Kachel schon mit dem Schatten zertifiziert? (dann weder Laden noch Rechnen) */
static inline int kachel_frei(const int16_t *s8, int32_t ng, int32_t nk, int32_t g0, int32_t gc, int32_t rows, const int16_t *esh,
                              const int16_t *mg)
{
   int32_t es = esh[g0];
   if (rows > 8 && esh[g0 + 1] < es) es = esh[g0 + 1];
   if (rows > 16 && esh[g0 + 2] < es) es = esh[g0 + 2];
   if (es == TS_ESH_UNBEKANNT || nk % TS_SK_KC) return 0;
   /* Grobstufe: Maximum ueber den ganzen k-Bereich (mg) — eine Pruefung statt nk/8; reicht sie nicht,
      entscheidet die Pruefung je Chunk. Beide sind hinreichend (mg >= jeder Chunkeintrag). */
   int32_t rmg = mg[g0];
   if (rows > 8 && mg[g0 + 1] > rmg) rmg = mg[g0 + 1];
   if (rows > 16 && mg[g0 + 2] > rmg) rmg = mg[g0 + 2];
   if (rmg + mg[gc] <= es - TS_SKIP_SCHWELLE) return 1;
   for (int32_t q = 0; q < nk; q += TS_SK_KC) {
      const int16_t *z = s8 + (size_t)(q / TS_SK_KC) * (size_t)ng;
      int32_t rm = z[g0];
      if (rows > 8 && z[g0 + 1] > rm) rm = z[g0 + 1];
      if (rows > 16 && z[g0 + 2] > rm) rm = z[g0 + 2];
      if (rm + z[gc] > es - TS_SKIP_SCHWELLE) return 0;
   }
   return 1;
}

/* Schatten der Spaltengruppen [c_von, c_bis) ungueltig machen (Elemente ohne Schattenpflege geaendert). */
static void schatten_verwerfen(const ts_skizze *skz, int32_t c_von, int32_t c_bis)
{
   if (!skz || c_von >= c_bis) return;
   for (int32_t gc = c_von / 8; gc <= (c_bis - 1) / 8 && gc < skz->ng; ++gc)
      for (int32_t g = 0; g < skz->ng; ++g) skz->esh[(size_t)gc * skz->ng + g] = TS_ESH_UNBEKANNT;
}

static void trail_v(int32_t n, double *ap, int32_t k0, int32_t k1, int32_t from, int32_t to, const int32_t *frow,
                    const ts_skizze *skz)
{
   if (from >= to || k0 >= k1) return;
   const int32_t nk = k1 - k0;
   const int16_t *s8 = (skz && nk >= SKIP_MIN_NK && k0 % TS_SK_KC == 0 && from % 8 == 0) ? skz->sk8 + (size_t)(k0 / TS_SK_KC) * (size_t)skz->ng : 0;
   const int16_t *ske = (skz && k0 % TS_SK_KC == 0) ? skz->sk + (size_t)(k0 / TS_SK_KC) * (size_t)n : 0;  /* Element-Skizze (Kopf) */
   const int32_t ng = skz ? skz->ng : 0;
   if (skz && !s8) schatten_verwerfen(skz, from, to);    /* dieser Aufruf pflegt den Schatten nicht */
   const size_t prb = (size_t)RB + 2 * TR;                 /* gepolsterte Zeilen je k (Lesen hinter rb erlaubt) */
   double *P  = (double *)aligned_alloc(64, sizeof(double) * prb * (size_t)nk);
   double *Cv = (double *)aligned_alloc(64, sizeof(double) * TC * (size_t)nk);
   double scratch[TR + 8];
   int16_t *mg = s8 ? (int16_t *)malloc(sizeof(int16_t) * (size_t)ng) : 0;     /* Gruppenmaximum ueber alle Chunks */
   if (!P || !Cv || (s8 && !mg)) { free(P); free(Cv); free(mg); schatten_verwerfen(skz, from, to); ts_chol_isa_scalar.trail(n, ap, k0, k1, from, to, frow, skz); return; }
   memset(scratch, 0, sizeof scratch);
   if (s8)
      for (int32_t g = from / 8; g < ng; ++g) {
         int32_t m = TS_E_NULL;
         for (int32_t q = 0; q + TS_SK_KC <= nk; q += TS_SK_KC) { const int32_t v = s8[(size_t)(q / TS_SK_KC) * (size_t)ng + g]; if (v > m) m = v; }
         mg[g] = (int16_t)m;
      }

   for (int32_t r0 = from; r0 < n; r0 += RB) {
      const int32_t r1 = imin(n, r0 + RB), rb = r1 - r0;
      const int32_t cend = imin(to, r1);                      /* Spalten mit Zeilen in diesem Block */
      if (from >= cend) break;
      const int32_t nch = (nk + TS_SK_KC - 1) / TS_SK_KC;
      unsigned char pq[nch], cq[nch];
      memset(pq, 0, sizeof pq);
      paket pk = { P, prb, pq, Cv, cq, ap, n, k0, r0, rb, 0, 0 };
      for (int32_t c2 = from; c2 < cend; c2 += TC) {
         const int32_t nc = imin(TC, cend - c2);
         double *colp[TC];
         for (int jj = 0; jj < TC; ++jj)
            colp[jj] = (jj < nc) ? ap + ts_col0(n, c2 + jj) - (c2 + jj) : scratch - 0;   /* scratch: Lesen/Schreiben maskiert */
         memset(cq, 0, sizeof cq);
         pk.c2 = c2; pk.nc = nc;
         /* Dreieckskopf: Zeilen [max(r0,c2), min(r1, c2+TC)) skalar, nur j <= i; L direkt aus ap.
            Chunkweise: erst das Blockzertifikat (Gruppenskizze der Diagonalgruppe gegen das frische
            ts_es-Minimum aller Kopfelemente), sonst je Element die Element-Skizze (ts_skip.h). */
         const int32_t ih0 = imax(r0, c2), ih1 = imin(r1, c2 + TC);
         if (ih0 < ih1) {
            const int32_t gk = c2 / 8;                         /* Kopfzeilen und -spalten liegen in Gruppe gk */
            int32_t esk = 0, frisch = 0;
            for (int32_t q = 0; q < nk; q += TS_SK_KC) {
               const int32_t qe = imin(q + TS_SK_KC, nk);
               if (s8 && qe - q == TS_SK_KC) {
                  if (!frisch) {
                     esk = 1 << 20;
                     for (int32_t i = ih0; i < ih1; ++i)
                        for (int jj = 0; jj < nc && c2 + jj <= i; ++jj) { const int32_t e = ts_es(colp[jj][i]); if (e < esk) esk = e; }
                     frisch = 1;
                  }
                  if (2 * s8[(size_t)(q / TS_SK_KC) * (size_t)ng + gk] <= esk - TS_SKIP_SCHWELLE) continue;
               }
               int geaendert = 0;
               for (int32_t i = ih0; i < ih1; ++i) {
                  if (frow && frow[i] - 1 >= k1) continue;
                  for (int jj = 0; jj < nc && c2 + jj <= i; ++jj) {
                     const int32_t c = c2 + jj;
                     double s = colp[jj][i];
                     if (ske && qe - q == TS_SK_KC) {
                        const int16_t *z = ske + (size_t)(q / TS_SK_KC) * (size_t)n;
                        if (z[i] + z[c] <= ts_es(s) - TS_SKIP_SCHWELLE) continue;
                     }
                     for (int32_t kk = q; kk < qe; ++kk) {
                        const double *lk = ap + ts_col0(n, k0 + kk) - (k0 + kk);
                        const double lck = lk[c];
                        if (lck != 0.0) s = fma(-lck, lk[i], s);
                     }
                     colp[jj][i] = s; geaendert = 1;
                  }
               }
               if (geaendert) frisch = 0;
            }
         }
         /* Vektorteil: Zeilen ab max(r0, c2+TC) */
         int16_t *esh = s8 ? skz->esh + (size_t)(c2 / 8) * (size_t)ng : 0;
         for (int32_t i = imax(r0, c2 + TC); i < r1; i += TR) {
            const int32_t rows = imin(TR, r1 - i);
            if (frow) {
               int32_t fmin = frow[i];
               for (int32_t t = 1; t < rows; ++t) if (frow[i + t] < fmin) fmin = frow[i + t];
               if (fmin - 1 >= k1) continue;                /* ganze Kachel: L(i,k) = 0 fuer alle k < k1 */
            }
            if (esh && kachel_frei(s8, ng, nk, i / 8, c2 / 8, rows, esh, mg)) continue;   /* nichts zu tun, Schatten bleibt gueltig */
            /* ungueltige Spalten muessen beim Laden/Speichern unschaedlich sein: scratch-Zeiger so legen,
               dass colp[jj] + i auf scratch zeigt */
            for (int jj = nc; jj < TC; ++jj) colp[jj] = scratch - i;
            kachel(rows, colp, i, i - r0, &pk, nk, s8, ng, i / 8, c2 / 8, nc, esh);
         }
      }
   }
   free(P); free(Cv); free(mg);
}

/* Skizze des Chunks ab Spalte q fuer die Zeilen ab q+8, je 8 Zeilen zugleich (Bitfeld-Exponenten). */
static void sketch_v(int32_t n, const double *ap, int32_t q, const ts_skizze *skz)
{
   const int32_t K = q / TS_SK_KC, i0 = q + TS_SK_KC;
   int16_t *sk = skz->sk + (size_t)K * (size_t)n;
   int16_t *s8 = skz->sk8 + (size_t)K * (size_t)skz->ng;
   const double *lk[TS_SK_KC];
   for (int t = 0; t < TS_SK_KC; ++t) lk[t] = ap + ts_col0(n, q + t) - (q + t);   /* lk[t][i] = L(i, q+t) */
   for (int32_t i = i0; i < n; i += 8) {                                         /* i0 ist ein Vielfaches von 8 */
      const __mmask8 mk = (__mmask8)(n - i >= 8 ? 0xFF : ((1u << (n - i)) - 1u));
      __m512i m = _mm512_set1_epi64(TS_E_NULL);
      for (int t = 0; t < TS_SK_KC; ++t) m = _mm512_max_epi64(m, e_vec(_mm512_maskz_loadu_pd(mk, lk[t] + i)));
      m = _mm512_mask_mov_epi64(_mm512_set1_epi64(TS_E_NULL), mk, m);           /* Zeilen hinter n zaehlen nicht */
      _mm512_mask_cvtepi64_storeu_epi16(sk + i, mk, m);                         /* vpmovqw: AVX-512F */
      s8[i / 8] = (int16_t)_mm512_reduce_max_epi64(m);
   }
}

const ts_chol_isa ts_chol_isa_avx512 = { axpy_v, scal_v, trail_v, sketch_v, "avx512" };

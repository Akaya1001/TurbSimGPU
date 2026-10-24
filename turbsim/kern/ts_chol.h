/* ts_chol.h — gemeinsame Deklarationen der bitgleichen Cholesky (dpptrf-Spezifikation).
 *
 * Spezifikation je Element (i,c), i >= c (verifiziert gegen OpenBLAS dpptrf_):
 *   s = A(i,c); fuer k = 0..c-1 AUFSTEIGEND: wenn L(c,k) != 0: s = fma(-L(c,k), L(i,k), s)
 *   L(c,c) = sqrt(s_cc)  (vorher s_cc <= 0 -> INFO = c+1);  r = 1.0/L(c,c);  L(i,c) = s_ic * r
 * Gepacktes 'L'-Format, spaltenweise: Spalte c (0-basiert) beginnt bei
 *   col0(c) = c*n - c*(c-1)/2, Element (i,c) liegt bei col0(c) + (i-c).
 */
#ifndef TS_CHOL_H
#define TS_CHOL_H
#include <stdint.h>
#include <stddef.h>

static inline size_t ts_col0(int32_t n, int32_t c) { return (size_t)c * (size_t)n - ((size_t)c * (size_t)(c - 1)) / 2; }

/* Rang-1-Baustein: y[i] = fma(-a, x[i], y[i]) fuer i = 0..m-1 (a != 0 vom Aufrufer gesichert). */
typedef void (*ts_axpy_fn)(int32_t m, const double *x, double *y, double a);
/* Skalierung: x[i] = x[i] * r */
typedef void (*ts_scal_fn)(int32_t m, double *x, double r);
/* Exponenten-Skizze fuer die zertifizierte No-op-Elimination (Regel in ts_skip.h), Chunks zu TS_SK_KC Spalten:
 *   sk [K*n  + i] = max_{k in Chunk K} ts_e(L(i,k))   fuer Zeilen i >= Chunkende (nur fertige Spalten)
 *   sk8[K*ng + g] = max ueber die Zeilen 8g..8g+7 davon  (ng = (n+7)/8)
 * Eintraege >= TS_E_OFFEN (auch das memset-Muster 0x7F7F) gelten als ungueltig: nie zertifiziert.
 * Ein Chunk K darf fuer Element (i,c) und Akkumulator s ausgelassen werden, wenn
 *   max_i sk + max_c sk <= ts_es(s) - TS_SKIP_SCHWELLE  (fuer alle Elemente der Kachel). */
#define TS_SK_KC 8
/* Schatten: esh[gc*ng + g] = untere Schranke von ts_es ueber die Akkumulatoren des 8x8-Blocks
 * (Zeilen 8g..8g+7, Spalten 8gc..8gc+7), gueltig, solange sich diese Elemente nicht geaendert haben.
 * TS_ESH_UNBEKANNT (memset-Muster 0x80) erlaubt nie ein Auslassen. Gepflegt nur im AVX-512-Kern
 * (Kacheln aus ganzen 8er-Gruppen); wer Elemente ohne Pflege aendert, setzt sie auf UNBEKANNT. */
#define TS_ESH_UNBEKANNT ((int16_t)(-32640))
typedef struct { int16_t *sk, *sk8, *esh; int32_t ng; } ts_skizze;

/* Nachlauf-Update: Spalten [from,to) mit k in [k0,k1) (aufsteigend), Zeilen i >= Spalte.
 * frow: 1-basierte erste Spalte != 0 je Zeile (oder NULL): Zeilen mit frow[i]-1 >= k1 haben
 * L(i,k) = 0 fuer alle k < k1 und werden ausgelassen (wertgleich; nur Nullvorzeichen koennen abweichen).
 * skz: Skizze (oder NULL = ohne No-op-Elimination); k0 ist dann ein Vielfaches von TS_SK_KC. */
typedef void (*ts_trail_fn)(int32_t n, double *ap, int32_t k0, int32_t k1, int32_t from, int32_t to, const int32_t *frow,
                            const ts_skizze *skz);
/* Skizze des Chunks ab Spalte q (Vielfaches von TS_SK_KC, Spalten q..q+7 fertig skaliert) fuer Zeilen >= q+8. */
typedef void (*ts_sketch_fn)(int32_t n, const double *ap, int32_t q, const ts_skizze *skz);

typedef struct { ts_axpy_fn axpy; ts_scal_fn scal; ts_trail_fn trail; ts_sketch_fn sketch; const char *name; } ts_chol_isa;

/* Implementierungen (je eine Uebersetzungseinheit) */
extern const ts_chol_isa ts_chol_isa_scalar;
extern const ts_chol_isa ts_chol_isa_avx2;
extern const ts_chol_isa ts_chol_isa_avx512;

/* Generische Hilfen fuer die Kerne */
#define TS_TRAIL_HEAD_ROWS 8      /* Dreiecks-Kopf eines 8-Spalten-Blocks wird skalar gerechnet */
#endif

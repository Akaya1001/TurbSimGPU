/* ts_skip.h — zertifizierte No-op-Regel fuer die dpptrf-Kette (eine Quelle fuer CPU und GPU).
 *
 * Eine FMA s' = fma(-a, b, s), deren gerundetes Ergebnis bitweise s ist, darf entfallen: die
 * Kette je Element bleibt identisch (Induktion ueber k). Hinreichend, nur Ganzzahlen:
 *
 *      ts_e(a) + ts_e(b) <= ts_es(s) - TS_SKIP_SCHWELLE        (Schwelle 56)
 *
 * Beweis. Fuer endliches x != 0 gilt |x| < 2^(ts_e(x)+1) (auch subnormal: Feld 0 -> -1023).
 *   s normal, |s| in [2^E, 2^(E+1)), ts_es = E: |ab| < 2^(E-54) = ulp(s)/4. Das ist weniger als die
 *     halbe Luecke zum Nachbarn auf beiden Seiten, auch wenn s eine Zweierpotenz ist (Luecke nach
 *     unten dort ulp(s)/2) -> RN(s - ab) = s.
 *   s subnormal oder +-0, ts_es = -1022: |ab| < 2^-1076 < 2^-1075 (halber Subnormalabstand)
 *     -> Ergebnis s, bei s = +-0 eine Null mit evtl. anderem Vorzeichen. Das ist erlaubt: Nullen
 *     wirken nur auf Nullen, V startet bei +0 und addiert nur.
 *   a oder b exakt Null (TS_E_NULL), der andere Faktor endlich: Produkt +-0 -> s bzw. Nullvorzeichen.
 *   Inf/NaN in a, b oder s: Summe bzw. Schwelle so gewaehlt, dass nie uebersprungen wird.
 * Gegenbeispiele gegen die naheliegenden Abkuerzungen (Recherche 14.10.2026):
 *   Schwelle 55 statt 56: a=0x1.fffffffffffffp-500, b=0x1.fffffffffffffp-555, s=0x1p-1000
 *     -> fma = 0x1.fffffffffffffp-1001 != s.
 *   reelle Regel fl(|a|*|b|)*2^55 <= |s|: a=2^-500, b=2^-575, s=0x1.0000000000001p-1022 (Produkt
 *     rundet in FP64 auf 0, fma ergibt 0x1p-1022).
 *
 * NUR Bitfeld-Extraktion verwenden: frexp/ilogb/v_frexp_exp haben eine andere Konvention
 * (floor(log2)+1) bzw. liefern fuer Inf/NaN 0 — gemischt verletzt das die Regel unbemerkt.
 *
 * Sketch je k-Chunk: Sk[i][K] = max_{k in K} ts_e(L(i,k)), erst geschrieben, wenn ALLE Spalten des
 * Chunks fertig skaliert sind (vorher TS_E_OFFEN; |L_ic| >= |s_ic|, weil r_c = 1/L_cc >= 1).
 * Zertifikat fuer Element (i,c) und Chunk K beim aktuellen s:
 *      Sk[i][K] + Sk[c][K] <= ts_es(s) - TS_SKIP_SCHWELLE
 * Dann sind alle k in K No-ops: s aendert sich im Chunk nicht, die Elementregel gilt fuer jedes k.
 */
#ifndef TS_SKIP_H
#define TS_SKIP_H
#include <stdint.h>
/* __builtin_memcpy statt memcpy: unter HIP ist memcpy aus <string.h> eine reine Host-Funktion. */

#if defined(__HIPCC__) || defined(__CUDACC__)
#define TS_SKIP_FN static inline __host__ __device__
#else
#define TS_SKIP_FN static inline
#endif

#define TS_E_NULL   (-2048)  /* exakte Null (+-0); unterscheidet sich von subnormal (-1023) */
#define TS_E_NAN    (4096)   /* Inf/NaN als Faktor: nie ueberspringen */
#define TS_E_OFFEN  (4096)   /* Sketch-Eintrag noch nicht gueltig */
#define TS_SKIP_SCHWELLE 56

/* Obere Exponentenschranke eines Faktors: |x| < 2^(ts_e(x)+1) fuer endliches x != 0. */
TS_SKIP_FN int32_t ts_e(double x)
{
   uint64_t u; __builtin_memcpy(&u, &x, sizeof u);
   const int32_t f = (int32_t)((u >> 52) & 0x7FF);
   if (f == 0) return (u << 1) ? -1023 : TS_E_NULL;
   if (f == 0x7FF) return TS_E_NAN;
   return f - 1023;
}

/* Untere Exponentenschranke des Akkumulators: |s| >= 2^ts_es(s) fuer normales s; -1022 fuer
 * subnormal/Null; Inf/NaN so klein, dass keine Summe darunter liegt. */
TS_SKIP_FN int32_t ts_es(double s)
{
   uint64_t u; __builtin_memcpy(&u, &s, sizeof u);
   const int32_t f = (int32_t)((u >> 52) & 0x7FF);
   if (f == 0x7FF) return -2 * TS_E_NAN;
   return (f ? f : 1) - 1023;
}

/* 1: fma(-a, b, s) liefert bitweise s (bzw. bei s = +-0 eine Null) — darf entfallen. */
TS_SKIP_FN int ts_skip(double a, double b, double s)
{
   return ts_e(a) + ts_e(b) <= ts_es(s) - TS_SKIP_SCHWELLE;
}
#endif

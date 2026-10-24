/* ts_kern.c — bitgleiche Bausteine fuer TurbSims Frequenzschleife (Stufe 1a).
 *
 * Alles hier ist dazu da, die Rechnung der Referenz (/usr/local/bin/turbsim,
 * v3.5.3, gfortran -O3 ohne -march) BITGLEICH zu reproduzieren, nur schneller.
 *
 * 1. Dist-Klassen: Das exp-Argument der Kohaerenz haengt je Frequenz nur vom
 *    Bitmuster von Dist ab (DistU = Dist/UHub ist eine Funktion davon). Eine
 *    Box hat nur K verschiedene Dist-Bitmuster (std 1240, W960 13 967) statt
 *    NPacked (0,75-9,4 Mio.). Je Frequenz genuegen also 2K exp-Auswertungen.
 * 2. Welche exp? gfortran hat die Schleife paarweise mit libmvec
 *    _ZGVbN2v_exp vektorisiert; nur das letzte Element I=N jeder Spalte mit
 *    UNGERADER Laenge N-J+1 laeuft ueber das skalare glibc-exp. Beide weichen
 *    bei ~43 % der Argumente um bis zu 3 ULP voneinander ab. Darum traegt
 *    jeder gepackte Eintrag seine Klasse UND seine Paritaet (Vektor/Skalar),
 *    und die Tabelle enthaelt beide Werte je Klasse. Wir rufen die ECHTEN
 *    glibc-Funktionen auf (keine Nachbauten).
 * 3. Argumentaufbau exakt in der Maschinenfolge der Referenz:
 *      x = Freq*DistU; y = InCohB*Dist; t = x*x + y*y; s = sqrt(t);
 *      a = InCDec*s; arg = -a
 *    Diese Datei MUSS mit -ffp-contract=off und ohne -ffast-math uebersetzt
 *    werden (keine FMA-Kontraktion, kein Umordnen).
 *
 * Aufrufkonvention: Fortran bind(C), alle Argumente per Referenz; Felder sind
 * die gepackten LAPACK-'L'-Felder (Indx = N*(J-1) - J*(J-1)/2 + I, 1-basiert,
 * I laeuft schnell).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emmintrin.h>

/* libmvec, SSE-Variante mit 2 Lanes (glibc >= 2.22). */
__m128d _ZGVbN2v_exp(__m128d);

static int vgl_double(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return (x < y) ? -1 : (x > y);
}

/* Klassen bilden.
 *   np, n       : NPacked, NPoints
 *   dist, distu : double[np] (Fortran-Felder, unveraendert uebernommen)
 *   cls         : int32[np], Ausgabe: 2*Klasse + Paritaet (Paritaet 1 = skalares exp)
 *   distk,distuk: double[np] Puffer, Ausgabe: Klassenrepraesentanten 0..k-1 (aufsteigend)
 *   k           : Ausgabe Klassenzahl
 * Dist >= +0, also ist die Ordnung der Werte die Ordnung der Bitmuster; verschiedene
 * Bitmuster sind verschiedene Werte. */
void ts_cls_build(const int32_t *np_, const int32_t *n_, const double *dist, const double *distu,
                  int32_t *cls, double *distk, double *distuk, int32_t *k_)
{
   const int64_t np = *np_;
   const int32_t n = *n_;
   double *tmp = (double *)malloc((size_t)np * sizeof(double));
   if (!tmp) { *k_ = -1; return; }
   memcpy(tmp, dist, (size_t)np * sizeof(double));
   qsort(tmp, (size_t)np, sizeof(double), vgl_double);
   int64_t k = 0;
   for (int64_t e = 0; e < np; ++e)
      if (k == 0 || tmp[e] != distk[k - 1]) distk[k++] = tmp[e];
   free(tmp);
   /* DistU-Repraesentant: der Wert aus dem Fortran-Feld beim ersten Treffer. */
   for (int64_t c = 0; c < k; ++c) distuk[c] = 0.0;
   int64_t e = 0;
   for (int32_t J = 1; J <= n; ++J) {
      const int32_t len = n - J + 1;
      const int32_t par_last = (len & 1) ? 1 : 0;     /* letztes Element skalar, wenn Spaltenlaenge ungerade */
      for (int32_t I = J; I <= n; ++I, ++e) {
         /* Binaersuche der Klasse */
         const double d = dist[e];
         int64_t lo = 0, hi = k - 1;
         while (lo < hi) { int64_t mid = (lo + hi) >> 1; if (distk[mid] < d) lo = mid + 1; else hi = mid; }
         const int32_t par = (I == n) ? par_last : 0;
         cls[e] = (int32_t)(2 * lo + par);
         if (distuk[lo] == 0.0) distuk[lo] = distu[e];
      }
   }
   *k_ = (int32_t)k;
}

/* Tabelle je Frequenz: tab[2c] = Vektor-exp(arg_c) (Lane 0 von _ZGVbN2v_exp),
 * tab[2c+1] = skalares exp(arg_c). */
void ts_exp_tab(const int32_t *k_, const double *distk, const double *distuk,
                const double *uhub_unused, const double *freq_, const double *incdec_, const double *incohb_,
                double *tab)
{
   (void)uhub_unused;
   const int32_t k = *k_;
   const double freq = *freq_, incdec = *incdec_, incohb = *incohb_;
   for (int32_t c = 0; c < k; ++c) {
      const double x = freq * distuk[c];
      const double y = incohb * distk[c];
      const double t = x * x + y * y;
      const double s = sqrt(t);
      const double a = incdec * s;
      const double arg = -a;
      __m128d v = _ZGVbN2v_exp(_mm_set1_pd(arg));
      tab[2 * c]     = _mm_cvtsd_f64(v);
      tab[2 * c + 1] = exp(arg);
   }
}

/* Gather: trh[e] = tab[cls[e]] fuer alle Eintraege ab Indx0 (1-basiert), dabei je Zeile I
 * die erste Spalte mit Eintrag != 0 bestimmen (frow[I-1], 1-basiert; 0 = keine). */
void ts_coh_fill(const int32_t *n_, const int32_t *indx0_, const int32_t *cls, const double *tab,
                 double *trh, int32_t *frow)
{
   const int32_t n = *n_;
   const int64_t e0 = (int64_t)(*indx0_) - 1;
   for (int32_t i = 0; i < n; ++i) frow[i] = 0;
   int64_t e = 0;
   for (int32_t J = 1; J <= n; ++J) {
      if (e + (n - J + 1) <= e0) { e += n - J + 1; continue; }   /* Spalte liegt vor dem Startindex */
      for (int32_t I = J; I <= n; ++I, ++e) {
         if (e < e0) continue;
         const double v = tab[cls[e]];
         trh[e] = v;
         if (v != 0.0 && frow[I - 1] == 0) frow[I - 1] = J;
      }
   }
}

/* Selbsttest-Hilfen fuer ctypes: einzelne Auswertungen. */
double ts_test_exp_vec(double x) { return _mm_cvtsd_f64(_ZGVbN2v_exp(_mm_set1_pd(x))); }
double ts_test_exp_scal(double x) { return exp(x); }

/* ======================================================================
 * Stufe 1b: bitgleiche Cholesky nach der dpptrf-Spezifikation (siehe ts_chol.h)
 * ====================================================================== */
#include "ts_chol.h"

#define TS_NB_TOP 256      /* Panelbreite auf oberster Ebene */
#define TS_NB_MIN 16       /* darunter: ungeblockte Rang-1-Fassung */

static const ts_chol_isa *ts_isa_waehlen(void)
{
   static const ts_chol_isa *isa = 0;
   if (isa) return isa;
   const char *w = getenv("TS_CHOL_ISA");
   if (w && !strcmp(w, "scalar")) isa = &ts_chol_isa_scalar;
   else if (w && !strcmp(w, "avx2")) isa = &ts_chol_isa_avx2;
   else if (w && !strcmp(w, "avx512")) isa = &ts_chol_isa_avx512;
   else if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma")) isa = &ts_chol_isa_avx512;
   else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) isa = &ts_chol_isa_avx2;
   else isa = &ts_chol_isa_scalar;
   return isa;
}

/* Ungeblockte Fassung fuer die Spalten [c0, c0+cb): je Spalte sqrt, Kehrwert, Skalierung,
 * dann Rang-1-Update der RESTLICHEN Panelspalten (Spalten ausserhalb des Panels bekommen
 * dieselben Updates spaeter ueber trail() — je Element bleibt k aufsteigend). */
static int32_t faktor_ungeblockt(const ts_chol_isa *isa, int32_t n, double *ap, int32_t c0, int32_t cb, const ts_skizze *skz)
{
   for (int32_t c = c0; c < c0 + cb; ++c) {
      double *colc = ap + ts_col0(n, c) - c;                   /* colc[i] = A(i,c) */
      const double d = colc[c];
      if (d <= 0.0) return c + 1;                               /* dpptrf: AJJ.LE.ZERO -> INFO = J (NaN laeuft durch) */
      const double L = sqrt(d);
      colc[c] = L;
      const double r = 1.0 / L;
      if (c + 1 < n) isa->scal(n - c - 1, colc + c + 1, r);     /* DSCAL mit ONE/AJJ */
      for (int32_t c2 = c + 1; c2 < c0 + cb; ++c2) {           /* DSPR, nur Panelspalten; Spalte mit x==0 ueberspringen */
         const double x = colc[c2];
         if (x != 0.0) isa->axpy(n - c2, colc + c2, ap + ts_col0(n, c2), x);
      }
   }
   /* Skizze der vollstaendigen Chunks dieses Blocks: alle ihre Spalten sind jetzt fertig skaliert.
      Blockanfaenge sind Vielfache von 8 (Panel 256, Teilung (cb/2)&~7); sonst bleibt die Skizze
      ungueltig und es wird nichts ausgelassen. */
   if (skz && c0 % TS_SK_KC == 0)
      for (int32_t q = c0; q + TS_SK_KC <= c0 + cb; q += TS_SK_KC) isa->sketch(n, ap, q, skz);
   /* Diese Spalten wurden ohne Schattenpflege geaendert (sie sind jetzt fertig und werden nicht mehr
      aktualisiert); den Schatten trotzdem verwerfen, damit er nie fuer veraenderte Elemente gilt. */
   if (skz)
      for (int32_t gc = c0 / 8; gc <= (c0 + cb - 1) / 8 && gc < skz->ng; ++gc)
         for (int32_t g = 0; g < skz->ng; ++g) skz->esh[(size_t)gc * skz->ng + g] = TS_ESH_UNBEKANNT;
   return 0;
}

static int32_t faktor_rekursiv(const ts_chol_isa *isa, int32_t n, double *ap, int32_t c0, int32_t cb, const int32_t *frow,
                               const ts_skizze *skz)
{
   if (cb <= TS_NB_MIN) return faktor_ungeblockt(isa, n, ap, c0, cb, skz);
   int32_t cb1 = (cb / 2) & ~7;
   if (cb1 < 8) cb1 = 8;
   int32_t info = faktor_rekursiv(isa, n, ap, c0, cb1, frow, skz);
   if (info) return info;
   isa->trail(n, ap, c0, c0 + cb1, c0 + cb1, c0 + cb, frow, skz);
   return faktor_rekursiv(isa, n, ap, c0 + cb1, cb - cb1, frow, skz);
}

/* Skizzenspeicher je Faden, ueber Frequenzen wiederverwendet (W960: 4,7 MB + 0,6 MB). */
static const ts_skizze *skizze_holen(int32_t n)
{
   static __thread int16_t *buf = 0;
   static __thread size_t cap = 0;
   static __thread ts_skizze skz;
   const size_t nch = (size_t)(n + TS_SK_KC - 1) / TS_SK_KC, ng = (size_t)(n + 7) / 8;
   const size_t len = nch * (size_t)n + nch * ng + ng * ng;
   if (len > cap) {
      free(buf);
      buf = (int16_t *)malloc(len * sizeof(int16_t));
      cap = buf ? len : 0;
      if (!buf) return 0;                                       /* ohne Skizze: exakt, nur langsamer */
   }
   const size_t lsk = nch * (size_t)n + nch * ng;
   memset(buf, 0x7F, lsk * sizeof(int16_t));                    /* 0x7F7F >= TS_E_OFFEN: ungueltig */
   memset(buf + lsk, 0x80, ng * ng * sizeof(int16_t));          /* Schatten: TS_ESH_UNBEKANNT */
   skz.sk = buf; skz.sk8 = buf + nch * (size_t)n; skz.esh = buf + lsk; skz.ng = (int32_t)ng;
   return &skz;
}

/* TS_CHOL_SKIP=0 schaltet die zertifizierte No-op-Elimination ab (Vorgabe: an). Beide Wege
 * liefern dieselben Werte; nur Vorzeichen exakter Nullen in L koennen sich unterscheiden. */
static int ts_skip_modus = -1;
static int ts_skip_an(void)
{
   if (ts_skip_modus < 0) { const char *e = getenv("TS_CHOL_SKIP"); ts_skip_modus = (e && e[0] == '0') ? 0 : 1; }
   return ts_skip_modus;
}
/* Fuer Tests (ctypes): Skip ein-/ausschalten, unabhaengig von der Umgebung. */
void ts_chol_skip_setzen(int32_t an) { ts_skip_modus = an ? 1 : 0; }

/* Fortran bind(C): ts_chol_spec(n, ap, frow, use_frow, info)
 *   ap   : gepacktes 'L' (NPacked Doubles), wird durch L ueberschrieben
 *   frow : int32[n], 1-basierte erste Spalte != 0 je Zeile (nur gelesen, wenn use_frow != 0)
 *   info : 0 ok; > 0: fuehrender Minor der Ordnung info nicht positiv definit (wie LAPACK) */
void ts_chol_spec(const int32_t *n_, double *ap, const int32_t *frow, const int32_t *use_frow, int32_t *info)
{
   const int32_t n = *n_;
   const ts_chol_isa *isa = ts_isa_waehlen();
   const int32_t *fr = (*use_frow) ? frow : 0;
   int32_t inf = 0;
   /* Testhaken: TS_CHOL_TESTFAIL=<k> meldet 'Minor der Ordnung k nicht positiv definit',
    * damit der Fortran-Fehlerpfad (kleinste Frequenz, Meldungstext) gepruft werden kann. */
   { static int tf = -2; if (tf == -2) { const char *e = getenv("TS_CHOL_TESTFAIL"); tf = e ? atoi(e) : -1; }
     if (tf > 0) { *info = tf; return; } }
   /* TS_CHOL_TESTFAIL_A10=<x>: Minor 2 scheitert, wenn A(2,1) < x — also fuer alle Frequenzen oberhalb einer
    * Schwelle (die Kohaerenz der Nachbarpunkte faellt mit f). Prueft, dass die kleinste davon gemeldet wird,
    * auch wenn ein Faden mit hohen Frequenzen zuerst scheitert. */
   { static double ta = -2.0; if (ta == -2.0) { const char *e = getenv("TS_CHOL_TESTFAIL_A10"); ta = e ? atof(e) : -1.0; }
     if (ta > 0.0 && n > 1 && ap[1] < ta) { *info = 2; return; } }
   const ts_skizze *skz = ts_skip_an() ? skizze_holen(n) : 0;
   for (int32_t c0 = 0; c0 < n && !inf; c0 += TS_NB_TOP) {
      const int32_t cb = (n - c0 < TS_NB_TOP) ? (n - c0) : TS_NB_TOP;
      inf = faktor_rekursiv(isa, n, ap, c0, cb, fr, skz);
      if (!inf && c0 + cb < n) isa->trail(n, ap, c0, c0 + cb, c0 + cb, n, fr, skz);
   }
   *info = inf;
}

const char *ts_chol_isa_name(void) { return ts_isa_waehlen()->name; }

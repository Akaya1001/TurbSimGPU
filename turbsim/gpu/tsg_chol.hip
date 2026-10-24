// tsg_chol.hip — bitgleiche Cholesky (dpptrf-Kette) auf der R9700, Stufe G1 (Plan 14.10.2026): dicht, ohne Skip.
//
// "Frequenz-Lanes": eine Wave (32 Lanes) rechnet dieselben Elemente fuer 32 Frequenzen, jede Lane ihre eigene
// Matrix. Damit ist jede Lane eine skalare Kette nach der Spezifikation (ts_chol.h) — trivial bitgleich — und alle
// Lanes treffen spaeter (Stufe G2) fast dieselben Skip-Entscheidungen (benachbarte Frequenzen haben fast dieselben
// Kohaerenzmatrizen; G0: maskierte Lanes sparen keine FP64-Takte, also nur wave-einheitlich auslassen).
//
// Spezifikation je Element (i,c), i >= c:  s = A(i,c); fuer k = 0..c-1 AUFSTEIGEND: wenn L(c,k) != 0:
//   s = fma(-L(c,k), L(i,k), s);  L(c,c) = sqrt(s_cc) (s_cc <= 0 -> INFO = c+1);  r = 1.0/L(c,c);  L(i,c) = s * r.
// Der Referenz-Skip (L(c,k) == 0) wird je Lane per Auswahl nachgebildet — so sind auch die Nullvorzeichen gleich.
//
// Speicher: Gruppe g (32 Frequenzen), gepacktes 'L' spaltenweise wie ts_col0, frequenzverschraenkt:
//   Element e (0-basiert) der Lane l liegt bei L[((size_t)g * np + e) * 32 + l]   (np = n(n+1)/2)
// Ablauf je Spaltenblock c0 = 0, 8, 16, ... (Breite 8, = Skizzen-Chunk von G2):
//   tsg_k_bulk : alle Zeilenkacheln i >= c0, Spalten [c0, c0+8): Updates k < c0 (Akkumulator aus A, k aufsteigend)
//   tsg_k_diag : 8x8-Diagonalblock faktorisieren (sqrt, Kehrwert, Rang-1 im Block), r[c] nach R
//   tsg_k_panel: Zeilenkacheln i >= c0+8: Updates k in [c0, c0+c) aufsteigend, dann * r[c]
// Je Element bleibt damit die Kette k = 0..c-1 aufsteigend, Akkumulator aus A, explizites fma, kein Split-k.
//
//
// Stufe G2 (zertifizierte No-op-Elimination, Regel ts_skip.h): Das Panel schreibt je Lane die Gruppen-Skizze
//   S[((K * ng) + g) * 32 + lane] = max ueber Zeilen 8g..8g+7 und Spalten des Chunks K von ts_e(L)
// (nur fertige Werte; Eintraege 0x7F7F = ungueltig). Der Bulk-Kern laesst einen Chunk K fuer seine 8x8-Kachel aus,
// wenn fuer ALLE 32 Lanes gilt: S[K][Zeilengruppe] + S[K][Spaltengruppe] <= frisches min ts_es(Akkumulatoren) - 56
// (G0: maskierte Lanes sparen nichts, also nur wave-einheitlich). Ausfuehren ist immer exakt; nur Auslassen braucht
// das Zertifikat. Das Minimum wird nach jedem ausgefuehrten Chunk neu bestimmt, bevor ein Chunk entfallen darf.
// Grobstufe: Sc[K8][g] = max ueber die 8 Chunks K8*8..K8*8+7 (64 Spalten); nur vollstaendige 64er-Bloecke werden
// gelesen. Eine Pruefung laesst dann 8 Chunks auf einmal fallen (die Pruefkette je Wave ist latenzgebunden).
// Zeilenstufe: Se[K][i] = max ueber die 8 Spalten des Chunks K von ts_e(L(i,.)) je Zeile. Faellt das Kachelzertifikat
// durch, prueft der Bulk-Kern die 8 Kachelzeilen einzeln (Se[K][Zeile] + S[K][Spaltengruppe] <= min ts_es der Zeile
// - 56, wave-einheitlich) und rechnet den Chunk nur fuer die nicht zertifizierten Zeilen (mittlere Frequenzen).
//
// Uebersetzen mit: hipcc -O3 --offload-arch=gfx1201 -ffp-contract=off -fno-fast-math -fno-gpu-approx-transcendentals
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "../kern/ts_skip.h"

#define TSG_L 32      // Lanes = Frequenzen je Gruppe
#define TSG_W 8       // Spaltenblock / Kachelbreite

__device__ __forceinline__ size_t tsg_col0(int n, int c) { return (size_t)c * (size_t)n - ((size_t)c * (size_t)(c - 1)) / 2; }

// Bulk-Kachel je Wave: 8 Zeilen x TSG_TC Spalten (eine Haelfte des 8er-Spaltenblocks). 8x4 statt 8x8 halbiert den
// Registerbedarf (keine Auslagerung in den Scratch-Speicher, mehr Waves zur Latenzverdeckung) und macht das Zertifikat
// feiner (32 statt 64 Elemente je Lane).
#define TSG_TC 4

// Kleinstes ts_es je Kachelzeile (er[r]) der gueltigen Akkumulatoren dieser Lane; Rueckgabe: Minimum ueber die Kachel.
// Keine gueltigen Elemente -> sehr gross.
__device__ __forceinline__ int tsg_es_zeilen(const double acc[TSG_W][TSG_TC], int er[TSG_W], int i0, int cb, int n, int w)
{
   int mt = 1 << 20;
#pragma unroll
   for (int r = 0; r < TSG_W; ++r) {
      int m = 1 << 20;
#pragma unroll
      for (int cc = 0; cc < TSG_TC; ++cc) {
         const int i = i0 + r, c = cb + cc;
         if (i < n && cc < w && i >= c) { const int e = ts_es(acc[r][cc]); m = e < m ? e : m; }
      }
      er[r] = m; mt = m < mt ? m : mt;
   }
   return mt;
}

#define TSG_ZEIL_SPIEL 16     // Zeilenstufe nur, wenn das Kachelzertifikat hoechstens so viele Binaden verfehlt (Aufwand)

// Updates k < c0 fuer die Kachel Zeilen [i0, i0+8) x Spalten [cb, cb+4) (cb = c0 oder c0+4). S/Sc/Se: Skizzen (nur SKIP).
// Grid: x = 2 * Zeilenkacheln (gerade: linke, ungerade: rechte Spaltenhaelfte), y = Gruppe. Eine Wave je Workgroup:
// gemessen schneller als 2/4/8 Waves je Workgroup (15.10.: früh fertige Waves warten dort auf ihre Nachbarn).
template <bool SKIP>
__global__ __launch_bounds__(TSG_L) void tsg_k_bulk(double *L, int n, size_t np, int c0, const int16_t *S, const int16_t *Sc,
                                                    const int16_t *Se)
{
   const int lane = threadIdx.x;
   const int i0 = c0 + TSG_W * (int)(blockIdx.x >> 1), cb = c0 + TSG_TC * (int)(blockIdx.x & 1);
   const int w = min(TSG_TC, n - cb);
   if (w <= 0) return;
   double *Lg = L + (size_t)blockIdx.y * np * TSG_L + lane;
   double acc[TSG_W][TSG_TC];                                     // [Zeile r][Spalte cc]
#pragma unroll
   for (int r = 0; r < TSG_W; ++r)
#pragma unroll
      for (int cc = 0; cc < TSG_TC; ++cc) {
         const int i = i0 + r, c = cb + cc;
         acc[r][cc] = (i < n && cc < w && i >= c) ? Lg[(tsg_col0(n, c) + (size_t)(i - c)) * TSG_L] : 0.0;
      }
   const int ng = (n + TSG_W - 1) / TSG_W, gz = i0 / TSG_W, gs = c0 / TSG_W;   // Spaltengruppe = ganzer 8er-Block
   const int16_t *Sg = SKIP ? S + (size_t)blockIdx.y * (size_t)ng * (size_t)ng * TSG_L + lane : nullptr;
   const int nc8 = (ng + 7) / 8;
   const int16_t *Scg = SKIP ? Sc + (size_t)blockIdx.y * (size_t)nc8 * (size_t)ng * TSG_L + lane : nullptr;
   const int16_t *Seg = SKIP ? Se + (size_t)blockIdx.y * (size_t)ng * (size_t)n * TSG_L + lane : nullptr;
   int es = 0, er[TSG_W];                                         // Kachel- und Zeilenminima von ts_es
   bool frisch = false;
   if (SKIP) { es = tsg_es_zeilen(acc, er, i0, cb, n, w); frisch = true; }
   for (int q = 0; q < c0; q += TSG_W) {                          // c0 ist ein Vielfaches von 8: nur volle Chunks
      if (SKIP && (q % 64) == 0 && q + 64 <= c0) {                  // Grobstufe: 8 Chunks auf einmal
         const int K8 = q / 64;
         const int grob = (int)Scg[((size_t)K8 * ng + gz) * TSG_L] + (int)Scg[((size_t)K8 * ng + gs) * TSG_L];
         if (__ballot(!(grob <= es - TS_SKIP_SCHWELLE)) == 0) {
            if (!frisch) { es = tsg_es_zeilen(acc, er, i0, cb, n, w); frisch = true; }
            if (__ballot(!(grob <= es - TS_SKIP_SCHWELLE)) == 0) { q += 64 - TSG_W; continue; }   // 64 No-ops
         }
      }
      unsigned zeilen = 0xFFu;                                      // wave-einheitlich: Zeilen, die rechnen
      if (SKIP) {
         const int K = q / TSG_W;
         const int cm = (int)Sg[((size_t)K * ng + gs) * TSG_L];
         const int schranke = (int)Sg[((size_t)K * ng + gz) * TSG_L] + cm;
         // Vorpruefung mit (evtl. veraltetem) Minimum entscheidet nur, ob sich das Neubestimmen lohnt
         if (__ballot(!(schranke <= es - TS_SKIP_SCHWELLE)) == 0) {
            if (!frisch) { es = tsg_es_zeilen(acc, er, i0, cb, n, w); frisch = true; }
            if (__ballot(!(schranke <= es - TS_SKIP_SCHWELLE)) == 0) continue;   // alle Lanes: 8 No-ops
         }
         // Zeilenstufe, nur wenn das Kachelzertifikat knapp verfehlt wurde; Zeilenminima vorher frisch machen
         if (__ballot(!(schranke <= es - TS_SKIP_SCHWELLE + TSG_ZEIL_SPIEL)) == 0) {
            if (!frisch) { es = tsg_es_zeilen(acc, er, i0, cb, n, w); frisch = true; }
            const int16_t *se = Seg + ((size_t)K * n + (size_t)i0) * TSG_L;
#pragma unroll
            for (int r = 0; r < TSG_W; ++r) {
               const bool gueltig = i0 + r < n;
               const int rs = gueltig ? (int)se[(size_t)r * TSG_L] : 0;
               if (!gueltig || __ballot(!(rs + cm <= er[r] - TS_SKIP_SCHWELLE)) == 0) zeilen &= ~(1u << r);
            }
            if (zeilen == 0) continue;                                  // jede Zeile fuer sich zertifiziert
         }
      }
      for (int k = q; k < q + TSG_W; ++k) {
         const size_t b = tsg_col0(n, k) - (size_t)k;             // L(j,k) bei b + j
         double li[TSG_W], lc[TSG_TC];
#pragma unroll
         for (int r = 0; r < TSG_W; ++r) li[r] = ((zeilen & (1u << r)) && i0 + r < n) ? Lg[(b + (size_t)(i0 + r)) * TSG_L] : 0.0;
#pragma unroll
         for (int cc = 0; cc < TSG_TC; ++cc) lc[cc] = (cc < w) ? Lg[(b + (size_t)(cb + cc)) * TSG_L] : 0.0;
#pragma unroll
         for (int r = 0; r < TSG_W; ++r) {
            if (!(zeilen & (1u << r))) continue;                    // zertifizierte Zeile: 4 No-ops je k entfallen
#pragma unroll
            for (int cc = 0; cc < TSG_TC; ++cc) {
               const double t = __builtin_fma(-lc[cc], li[r], acc[r][cc]);
               acc[r][cc] = (lc[cc] != 0.0) ? t : acc[r][cc];       // Referenz-Skip bei L(c,k) == 0
            }
         }
      }
      frisch = false;
   }
#pragma unroll
   for (int r = 0; r < TSG_W; ++r)
#pragma unroll
      for (int cc = 0; cc < TSG_TC; ++cc) {
         const int i = i0 + r, c = cb + cc;
         if (i < n && cc < w && i >= c) Lg[(tsg_col0(n, c) + (size_t)(i - c)) * TSG_L] = acc[r][cc];
      }
}

// Diagonalblock [c0, c0+w): wie faktor_ungeblockt (ts_kern.c), nur auf den Blockzeilen. r[c] -> R[(g*8 + c)*32 + lane].
__global__ __launch_bounds__(TSG_L) void tsg_k_diag(double *L, int n, size_t np, int c0, double *R, int *info)
{
   const int lane = threadIdx.x, g = (int)blockIdx.x;
   double *Lg = L + (size_t)g * np * TSG_L + lane;
   const int w = min(TSG_W, n - c0);
   double s[TSG_W][TSG_W];
#pragma unroll
   for (int i = 0; i < TSG_W; ++i)
#pragma unroll
      for (int c = 0; c < TSG_W; ++c)
         s[i][c] = (i < w && c <= i) ? Lg[(tsg_col0(n, c0 + c) + (size_t)(i - c)) * TSG_L] : 0.0;
   int inf = info[g * TSG_L + lane];
#pragma unroll
   for (int c = 0; c < TSG_W; ++c) {
      if (c < w) {
         const double d = s[c][c];
         if (d <= 0.0 && inf == 0) inf = c0 + c + 1;                // dpptrf: AJJ.LE.ZERO -> INFO (NaN laeuft durch)
         const double Lcc = sqrt(d);
         s[c][c] = Lcc;
         const double r = 1.0 / Lcc;
         R[((size_t)g * TSG_W + c) * TSG_L + lane] = r;
#pragma unroll
         for (int i = c + 1; i < TSG_W; ++i) s[i][c] = s[i][c] * r;
#pragma unroll
         for (int c2 = c + 1; c2 < TSG_W; ++c2) {
            const double x = s[c2][c];
#pragma unroll
            for (int i = c2; i < TSG_W; ++i) {
               const double t = __builtin_fma(-x, s[i][c], s[i][c2]);
               s[i][c2] = (x != 0.0) ? t : s[i][c2];
            }
         }
      }
   }
   info[g * TSG_L + lane] = inf;
#pragma unroll
   for (int i = 0; i < TSG_W; ++i)
#pragma unroll
      for (int c = 0; c < TSG_W; ++c)
         if (i < w && c <= i) Lg[(tsg_col0(n, c0 + c) + (size_t)(i - c)) * TSG_L] = s[i][c];
}

// Zeilenkacheln unterhalb des Diagonalblocks: Updates k in [c0, c0+c) aufsteigend, dann * r[c].
// Mit S != nullptr und vollem Block (w == 8) schreibt jede Kachel die Gruppen-Skizze ihrer 8 Zeilen fuer Chunk c0/8.
__global__ __launch_bounds__(TSG_L) void tsg_k_panel(double *L, int n, size_t np, int c0, const double *R, int16_t *S, int16_t *Sc,
                                                     int16_t *Se)
{
   const int lane = threadIdx.x, g = (int)blockIdx.y;
   double *Lg = L + (size_t)g * np * TSG_L + lane;
   const int w = min(TSG_W, n - c0);
   const int i0 = c0 + TSG_W + TSG_W * (int)blockIdx.x;
   double ld[TSG_W][TSG_W], rr[TSG_W];                            // L(c0+c, c0+k), k < c; r[c]
#pragma unroll
   for (int c = 0; c < TSG_W; ++c) {
      rr[c] = (c < w) ? R[((size_t)g * TSG_W + c) * TSG_L + lane] : 0.0;
#pragma unroll
      for (int k = 0; k < TSG_W; ++k)
         ld[c][k] = (c < w && k < c) ? Lg[(tsg_col0(n, c0 + k) + (size_t)(c - k)) * TSG_L] : 0.0;
   }
   int emax = TS_E_NULL;                                          // Skizze: max ts_e ueber Kachelzeilen x Blockspalten
#pragma unroll
   for (int r = 0; r < TSG_W; ++r) {
      const int i = i0 + r;
      if (i >= n) break;
      double zl[TSG_W];                                           // fertige L(i, c0+c) dieser Zeile
      int ezeile = TS_E_NULL;                                     // Zeilenskizze: max ts_e ueber die Blockspalten
#pragma unroll
      for (int c = 0; c < TSG_W; ++c) {
         if (c < w) {
            const size_t e = tsg_col0(n, c0 + c) + (size_t)(i - c0 - c);
            double s = Lg[e * TSG_L];
#pragma unroll
            for (int k = 0; k < TSG_W; ++k)
               if (k < c) {
                  const double t = __builtin_fma(-ld[c][k], zl[k], s);
                  s = (ld[c][k] != 0.0) ? t : s;
               }
            s = s * rr[c];
            zl[c] = s;
            Lg[e * TSG_L] = s;
            const int ev = ts_e(s); emax = ev > emax ? ev : emax; ezeile = ev > ezeile ? ev : ezeile;
         }
      }
      if (Se && w == TSG_W)
         Se[((size_t)g * ((n + TSG_W - 1) / TSG_W) * n + (size_t)(c0 / TSG_W) * n + (size_t)i) * TSG_L + lane] = (int16_t)ezeile;
   }
   if (S && w == TSG_W) {
      const int ng = (n + TSG_W - 1) / TSG_W, K = c0 / TSG_W, nc8 = (ng + 7) / 8;
      S[((size_t)g * ng * ng + (size_t)K * ng + (size_t)(i0 / TSG_W)) * TSG_L + lane] = (int16_t)emax;
      // Grobstufe: erster Chunk eines 64er-Blocks ueberschreibt, die weiteren nehmen das Maximum (gleiche Kachel,
      // aufeinanderfolgende Launches -> kein Wettlauf). Gelesen wird nur, wenn alle 8 Chunks fertig sind.
      int16_t *sc = Sc + ((size_t)g * nc8 * ng + (size_t)(K / 8) * ng + (size_t)(i0 / TSG_W)) * TSG_L + lane;
      *sc = (K % 8 == 0 || *sc < emax) ? (int16_t)emax : *sc;
   }
}

// Groesse der Gruppen-Skizze (fein ng*ng + grob ceil(ng/8)*ng Eintraege je Lane und Gruppe) in Bytes fuer B Gruppen.
static size_t tsg_skizze_fein(int n, int B) { const size_t ng = (size_t)(n + TSG_W - 1) / TSG_W; return ng * ng * TSG_L * sizeof(int16_t) * (size_t)B; }
static size_t tsg_skizze_grob(int n, int B) { const size_t ng = (size_t)(n + TSG_W - 1) / TSG_W; return ((ng + 7) / 8) * ng * TSG_L * sizeof(int16_t) * (size_t)B; }
static size_t tsg_skizze_zeile(int n, int B) { const size_t ng = (size_t)(n + TSG_W - 1) / TSG_W; return ng * (size_t)n * TSG_L * sizeof(int16_t) * (size_t)B; }
static size_t tsg_skizze_bytes(int n, int B) { return tsg_skizze_fein(n, B) + tsg_skizze_grob(n, B) + tsg_skizze_zeile(n, B); }

// Host: B Gruppen zu je 32 Frequenzen in dL (frequenzverschraenkt), n Punkte. dR: B*8*32 doubles, dinfo: B*32 int
// (vorher 0). dS: Gruppen-Skizze (tsg_skizze_bytes) oder nullptr = dicht ohne Auslassen. Je Spaltenblock drei Kernel.
static void tsg_chol(double *dL, int n, int B, double *dR, int *dinfo, int16_t *dS, hipStream_t stream)
{
   const size_t np = (size_t)n * (size_t)(n + 1) / 2;
   int16_t *dSc = dS ? (int16_t *)((char *)dS + tsg_skizze_fein(n, B)) : nullptr;
   int16_t *dSe = dS ? (int16_t *)((char *)dS + tsg_skizze_fein(n, B) + tsg_skizze_grob(n, B)) : nullptr;
   if (dS) (void)hipMemsetAsync(dS, 0x7F, tsg_skizze_bytes(n, B), stream);   // 0x7F7F >= TS_E_OFFEN: ungueltig
   for (int c0 = 0; c0 < n; c0 += TSG_W) {
      if (c0 > 0) {
         dim3 gb((unsigned)(2 * ((n - c0 + TSG_W - 1) / TSG_W)), (unsigned)B);   // zwei Spaltenhaelften je Zeilenkachel
         if (dS) tsg_k_bulk<true><<<gb, TSG_L, 0, stream>>>(dL, n, np, c0, dS, dSc, dSe);
         else    tsg_k_bulk<false><<<gb, TSG_L, 0, stream>>>(dL, n, np, c0, nullptr, nullptr, nullptr);
      }
      tsg_k_diag<<<B, TSG_L, 0, stream>>>(dL, n, np, c0, dR, dinfo);
      if (n - c0 - TSG_W > 0) {
         dim3 gp((unsigned)((n - c0 - TSG_W + TSG_W - 1) / TSG_W), (unsigned)B);
         tsg_k_panel<<<gp, TSG_L, 0, stream>>>(dL, n, np, c0, dR, dS, dSc, dSe);
      }
   }
}
static void tsg_chol_dicht(double *dL, int n, int B, double *dR, int *dinfo, hipStream_t stream)
{
   tsg_chol(dL, n, B, dR, dinfo, nullptr, stream);
}

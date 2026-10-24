// tsg_host.hip — die ganze IEC-Frequenzschleife von TurbSim auf der R9700 (Stufe G3, Plan 14.10.2026), bitgleich.
//
// Je Stapel aus B Gruppen zu 32 aufeinanderfolgenden Frequenzen ("Frequenz-Lanes", tsg_chol.hip):
//   CPU (ts_gpu_prep.c): exp-Tabellen (echte glibc/libmvec), sincos der Phasen (glibc), S-Werte je Punktklasse
//   GPU: Gather A = tab[cls] -> Cholesky mit zertifiziertem Auslassen -> S-Faktoren -> H2Coeffs -> V-Scheiben
//   CPU: V-Scheiben in V(2f-1:2f, i, c) einsortieren (je Punkt ein zusammenhaengender Block)
// Die Vorbereitung des naechsten Stapels laeuft auf der CPU, waehrend die GPU den aktuellen rechnet.
//
// Rechenvorschrift wie die Referenz (TSsubs.f90 Coh2H + H2Coeffs, v3.5.3), je Element unveraendert:
//   H(i,j) = L(i,j) * SQRT(SQRT(ABS(S_i*S_j)))  (i > j),   H(j,j) = L(j,j) * SQRT(ABS(S_j))
//   V_i = (+0) + H(i,1)*cos(phi_1) + ... + H(i,i)*cos(phi_i)   (j aufsteigend, mul und add getrennt), ebenso sin.
// sqrt, mul, add, fma sind auf gfx1201 korrekt gerundet (T0.2, G0); uebersetzt mit -ffp-contract=off -fno-fast-math.
//
// C-Schnittstelle (Fortran bind(C), alles per Referenz): tsg_iec_lauf(...), siehe unten. Rueckgabe ok = 0: GPU nicht
// verfuegbar oder abgelehnt -> der Aufrufer rechnet den CPU-Pfad (gleiche Bits). fehl_ifreq > 0: kleinste Frequenz mit
// nicht positiv definiter Matrix; V ist fuer alle kleineren Frequenzen gefuellt, der Aufrufer rechnet diese eine
// Frequenz auf dem CPU-Pfad nach (Fehlertext byteweise wie die Referenz).
//
// Umgebung: TS_GPU=0 schaltet ab; TS_GPU_B=<B> Gruppen je Stapel (Vorgabe: so viele, wie ins VRAM-Budget passen,
// hoechstens 8); TS_GPU_VRAM_RESERVE_GB (Vorgabe 3); TS_GPU_SKIP=0 rechnet dicht (Pruefzwecke); TS_GPU_INFO=1 Bericht.
// (Ein HIP-Graph ueber die ~1 700 Starts je Stapel brachte am 15.10. nichts: die Luecken entstehen auf der GPU.)
// Schonbetrieb (der Rechner haengt bei minutenlanger Volllast, 14.10.): TS_GPU_CU=<n> rechnet nur auf n der 64 CUs
// (Stream-CU-Maske, paarweise je WGP) — weniger Leistung und Waerme, entsprechend laenger; TS_GPU_PAUSE=<p> (0..0,9)
// laesst die GPU nach jedem Stapel den Zeitanteil p ruhen (ohne Ueberlappung). Beides aendert keine Rechnung.
// Nur-GPU-Betrieb (gpu_windfelder.py, 17.10.): TS_GPU=nur sperrt den CPU-Pfad — kann die GPU nicht rechnen, meldet
// tsg_pflicht() das dem Fortran, und TurbSim bricht mit Fehler ab, statt die Schleife auf der CPU zu rechnen.
// TS_GPU_VRAM_MAX_GB=<g> deckelt das VRAM dieses Prozesses (mehrere TurbSim teilen sich die Karte).
// Waermebremse: TS_GPU_KANTE_MAX=<Grad C> (Kantentemperatur, hwmon "edge", Wiederanlauf TS_GPU_KANTE_WIEDER, Vorgabe -5)
// und TS_GPU_TJ_MAX=<Grad C> (Sperrschicht "junction", Wiederanlauf TS_GPU_TJ_WIEDER, Vorgabe -8): vor jedem Stapel lesen
// und ab der Grenze warten, bis beide wieder darunter sind. TS_GPU_STAPEL_S=<s> waehlt die
// Stapelgroesse laufend so, dass ein Stapel etwa s Sekunden GPU-Zeit braucht (Reaktionszeit der Bremse; bei tiefen
// Frequenzen dauert ein voller Stapel W960 sonst 8-15 s). Gruppen bleiben an Vielfachen von 32 Frequenzen ausgerichtet,
// jede Lane ist ihre eigene Kette: Stapelgroesse und Wartezeiten aendern kein Bit.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <string>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "tsg_chol.hip"

extern "C" {
void tsgp_exp(int32_t K, const double *distk, const double *distuk, double uhub, const double *freq, int32_t nl,
              double incdec, double incohb, double *tab, double *tmp);
void tsgp_phasen(int32_t n, const double *phase, int32_t ldp, int32_t nfp, const int32_t *ifreq, int32_t nl, int32_t ivec,
                 double *cs, double *sn);
void tsgp_s(int32_t m, const int32_t *rep, const double *S, int32_t nfs, int32_t nps, const int32_t *ifreq, int32_t nl,
            int32_t ivec, double *sv);
int32_t tsgp_klassen(int32_t n, const double *S, int32_t nfs, int32_t nps, int32_t nfreq, int32_t ivec0, int32_t ncomp,
                     int32_t *kls, int32_t *rep, int32_t maxm);
}

#define TSG_MAXM 256          // hoechstens so viele S-Klassen (Hoehen); sonst CPU-Pfad

// A(e) der Lane l in Gruppe g = tab[g][cls[e]][l]
__global__ __launch_bounds__(256) void tsg_k_gather(double *L, size_t np, const int32_t *cls, const double *tab, int K2)
{
   const size_t e = (size_t)blockIdx.x * 8 + threadIdx.x / TSG_L;
   const int lane = threadIdx.x % TSG_L, g = (int)blockIdx.y;
   if (e >= np) return;
   L[((size_t)g * np + e) * TSG_L + lane] = tab[((size_t)g * K2 + cls[e]) * TSG_L + lane];
}

// S-Faktoren je Klassenpaar: Ft[g][c][a][b][l] = SQRT(SQRT(ABS(S_a*S_b))), Fd[g][c][a][l] = SQRT(ABS(S_a)).
__global__ __launch_bounds__(TSG_L) void tsg_k_faktoren(const double *sv, double *Ft, double *Fd, int m, int ncomp)
{
   const int lane = threadIdx.x, a = (int)blockIdx.x, gc = (int)blockIdx.y;          // gc = g*ncomp + c
   const double *s = sv + (size_t)gc * m * TSG_L;
   const double sa = s[(size_t)a * TSG_L + lane];
   Fd[((size_t)gc * m + a) * TSG_L + lane] = sqrt(fabs(sa));
   for (int b = 0; b < m; ++b) {
      const double p = sa * s[(size_t)b * TSG_L + lane];                                 // S_I * S_J (kommutativ)
      Ft[(((size_t)gc * m + a) * m + b) * TSG_L + lane] = sqrt(sqrt(fabs(p)));
   }
}

// H2Coeffs: je Wave eine Zeile i, je Lane eine Frequenz, alle Komponenten gemeinsam (L einmal gelesen).
// Vh[c][i][2*(g*32+l)] = Real-, [+1] = Imaginaerteil.
__global__ __launch_bounds__(TSG_L) void tsg_k_h2c(const double *L, size_t np, int n, const int32_t *kls, int m,
                                                   const double *Ft, const double *Fd, const double *cs, const double *sn,
                                                   int ncomp, double *Vh, int B)
{
   const int lane = threadIdx.x, i = (int)blockIdx.x, g = (int)blockIdx.y;
   const double *Lg = L + (size_t)g * np * TSG_L + lane;
   const int ki = kls[i];
   double vr[3] = {0.0, 0.0, 0.0}, vi[3] = {0.0, 0.0, 0.0};                         // V startet bei +0
   for (int j = 0; j <= i; ++j) {
      const double lv = Lg[(tsg_col0(n, j) + (size_t)(i - j)) * TSG_L];
      const int kj = kls[j];
      for (int c = 0; c < ncomp; ++c) {
         const size_t gc = (size_t)g * ncomp + c;
         const double F = (i == j) ? Fd[(gc * m + kj) * TSG_L + lane] : Ft[((gc * m + ki) * m + kj) * TSG_L + lane];
         const double h = lv * F;                                                        // Coh2H: TRH = TRH * Faktor
         const size_t pj = (gc * n + j) * TSG_L + lane;
         vr[c] = vr[c] + h * cs[pj];                                                     // H2Coeffs: mul, dann add
         vi[c] = vi[c] + h * sn[pj];
      }
   }
   for (int c = 0; c < ncomp; ++c) {
      const size_t o = ((size_t)c * n + i) * (size_t)(2 * TSG_L * B) + 2 * ((size_t)g * TSG_L + lane);
      Vh[o] = vr[c]; Vh[o + 1] = vi[c];
   }
}

#define HCK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "TurbSim-GPU: HIP-Fehler %s (%s:%d)\n", hipGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)

namespace {
bool pflicht_gesetzt() { const char *e = getenv("TS_GPU"); return e && (strncmp(e, "nur", 3) == 0 || strncmp(e, "NUR", 3) == 0); }

double env_zahl(const char *name, double vorgabe) { const char *e = getenv(name); return (e && *e) ? atof(e) : vorgabe; }

// Waermebremse aus hwmon (Pfad ueber die PCI-Adresse des HIP-Geraets). Zwei Grenzen, jede abschaltbar (<= 0: aus):
//   Kante (TS_GPU_KANTE_MAX/WIEDER): traege, zeigt die Durchwaermung — die eigentliche Regelgroesse. Der Hotspot folgt der
//     Leistung binnen ~1 s (18.10.: bei 250 W rund 28 C ueber der Kante) und faellt in jeder Pause sofort; eine Grenze
//     nur auf ihn laesst die Karte zwischen 66 und 96 C pendeln (Dauertest 18.10.).
//   Sperrschicht/Tj (TS_GPU_TJ_MAX/WIEDER): zusaetzlicher Riegel vor jedem Stapel.
// Ausfallsicher: ist ein eingestellter Sensor nicht lesbar, wird nicht ungebremst gerechnet, sondern nach jedem Stapel so
// lange geruht, wie er gerechnet hat (50 % Tastverhaeltnis); eine laufende Pause wird bei Lesefehlern fortgesetzt.
struct Bremse {
   std::string tj_pfad, kante_pfad;  // .../hwmon/hwmonN/tempK_input mit Label "junction" bzw. "edge"
   double max = 0, wieder = 0;       // Tj in Grad C
   double kmax = 0, kwieder = 0;     // Kante in Grad C
   double warte_max = 900;           // laengste einzelne Pause in s (TS_GPU_TJ_WARTE_MAX_S), danach Fehler
   double gewartet = 0, tj_hoechst = -1, kante_hoechst = -1, ersatz_s = 0; int pausen = 0, lesefehler = 0; bool gewarnt = false;

   bool an() const { return max > 0 || kmax > 0; }
   static double lesen(const std::string &pfad) {
      FILE *f = pfad.empty() ? nullptr : fopen(pfad.c_str(), "r");
      if (!f) return -1;
      long v = 0; const bool ok = fscanf(f, "%ld", &v) == 1; fclose(f);
      return ok ? v / 1000.0 : -1;
   }
   // Messwerte der eingestellten Grenzen; false, wenn einer davon nicht lesbar ist
   bool messen(double &tj, double &kante) const {
      tj = max > 0 ? lesen(tj_pfad) : 0; kante = kmax > 0 ? lesen(kante_pfad) : 0;
      return tj >= 0 && kante >= 0;
   }
   bool zu_heiss(double tj, double kante) const { return (max > 0 && tj >= max) || (kmax > 0 && kante >= kmax); }
   bool kuehl(double tj, double kante) const { return (max <= 0 || tj <= wieder) && (kmax <= 0 || kante <= kwieder); }
   // nahe an einer Grenze: hoechstens ein Stapel im Flug, damit eine Pause sofort wirkt
   bool nah() const {
      if (!an()) return false;
      double tj, kante;
      if (!messen(tj, kante)) return true;
      return (max > 0 && tj >= max - 5.0) || (kmax > 0 && kante >= kmax - 2.0);
   }
   void suchen(const hipDeviceProp_t &p) {
      char basis[96];
      snprintf(basis, sizeof basis, "/sys/bus/pci/devices/%04x:%02x:%02x.0/hwmon", p.pciDomainID, p.pciBusID, p.pciDeviceID);
      DIR *d = opendir(basis);
      if (!d) return;
      for (dirent *e; (e = readdir(d)) != nullptr;) {
         if (strncmp(e->d_name, "hwmon", 5) != 0) continue;
         for (int k = 1; k <= 8; ++k) {
            const std::string stamm = std::string(basis) + "/" + e->d_name + "/temp" + std::to_string(k);
            FILE *f = fopen((stamm + "_label").c_str(), "r");
            if (!f) continue;
            char lab[32] = {0};
            if (fgets(lab, sizeof lab, f)) {
               if (strncmp(lab, "junction", 8) == 0 && tj_pfad.empty()) tj_pfad = stamm + "_input";
               if (strncmp(lab, "edge", 4) == 0 && kante_pfad.empty()) kante_pfad = stamm + "_input";
            }
            fclose(f);
         }
      }
      closedir(d);
   }
   // Vor einem Stapel (die GPU ist dann leer oder rechnet hoechstens einen Stapel). letzter_s: GPU-Zeit des letzten
   // Stapels (fuer den Ersatzbetrieb ohne Sensor). Rueckgabe false: Pause dauerte laenger als warte_max.
   bool warten(double letzter_s) {
      if (!an()) return true;
      double tj, kante;
      if (!messen(tj, kante)) {                                  // kein Sensor: 50 % Tastverhaeltnis statt ungebremst
         if (!gewarnt) fprintf(stderr, "TurbSim-GPU: Waermebremse ohne Sensor (Tj %s, Kante %s) -> Ersatzbetrieb mit 50 %% Pause\n",
                               tj_pfad.empty() ? "fehlt" : tj_pfad.c_str(), kante_pfad.empty() ? "fehlt" : kante_pfad.c_str());
         gewarnt = true; ++lesefehler;
         const double ruhe = std::min(std::max(letzter_s, 0.25), 30.0);
         std::this_thread::sleep_for(std::chrono::duration<double>(ruhe));
         ersatz_s += ruhe;
         return true;
      }
      if (max > 0 && tj > tj_hoechst) tj_hoechst = tj;
      if (kmax > 0 && kante > kante_hoechst) kante_hoechst = kante;
      if (!zu_heiss(tj, kante)) return true;
      ++pausen;
      const auto t0 = std::chrono::steady_clock::now();
      double fehl_s = 0, gemeldet = 0;
      for (;;) {
         std::this_thread::sleep_for(std::chrono::milliseconds(250));
         const bool ok = messen(tj, kante);
         const double dauer = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
         if (ok) { fehl_s = 0; if (kuehl(tj, kante)) break; }
         else if ((fehl_s += 0.25) >= 20.0) break;               // Sensor weg: nach 20 s Ruhe weiter (GPU war leer)
         if (dauer - gemeldet >= 30.0) {
            gemeldet = dauer;
            fprintf(stderr, "TurbSim-GPU: Waermebremse wartet seit %.0f s (Tj %.0f C, Kante %.0f C)\n", dauer, tj, kante);
         }
         if (dauer > warte_max) {
            fprintf(stderr, "TurbSim-GPU: Waermebremse: Karte %.0f s nicht abgekuehlt (Tj %.0f C, Kante %.0f C; fremde Last?)\n",
                    dauer, tj, kante);
            gewartet += dauer;
            return false;
         }
      }
      gewartet += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      return true;
   }
};

struct Puffer {                      // ein Satz Host-(pinned) und Geraetepuffer je Stapel
   double *h_tab = nullptr, *h_cs = nullptr, *h_sn = nullptr, *h_sv = nullptr, *h_vh = nullptr; int *h_info = nullptr;
   double *d_tab = nullptr, *d_cs = nullptr, *d_sn = nullptr, *d_sv = nullptr, *d_vh = nullptr; int *d_info = nullptr;
   int f0 = 0, nl = 0, b = 0;        // erste Frequenz (0-basiert), gueltige Lanes, Gruppen dieses Stapels (<= B)
   hipEvent_t start = nullptr, fertig = nullptr;
};

struct Lauf {
   int n, nfreq, K, m, ncomp, ivec, B; size_t np;   // B: Gruppen, fuer die die Puffer angelegt sind (Hoechstwert)
   const double *distk, *distuk, *freq, *S, *phase; double uhub, incdec, incohb;
   int nfs, nps, ldp, nfp; double *V; int ldv, npv;
   std::vector<int32_t> kls, rep;
   double *dL = nullptr, *dR = nullptr, *dFt = nullptr, *dFd = nullptr; int16_t *dS = nullptr; int32_t *dcls = nullptr, *dkls = nullptr;
   Puffer p[2];
   hipStream_t st = nullptr;
   bool skip = true;
   double stapel_s = 0;              // Ziel-GPU-Zeit je Stapel (TS_GPU_STAPEL_S), 0 = immer B Gruppen
   double t_gruppe = -1;             // gemessene GPU-Sekunden je Gruppe (letzter fertiger Stapel), < 0: noch keine
   double gpu_summe = 0, t_letzter = 0; int stapel = 0, gruppen = 0;

   size_t n_tab(int b) const { return (size_t)b * 2 * K * TSG_L; }
   size_t n_ph(int b) const { return (size_t)b * ncomp * n * TSG_L; }
   size_t n_sv(int b) const { return (size_t)b * ncomp * m * TSG_L; }
   size_t n_vh() const { return (size_t)ncomp * n * 2 * TSG_L * B; }   // Zeilenabstand 2*32*B, unabhaengig von b

   // Gruppen fuer den naechsten Stapel ab Frequenz f0: B, oder so viele, wie in stapel_s Sekunden passen
   int stapel_b(int f0) const {
      const int rest = (nfreq - f0 + TSG_L - 1) / TSG_L;
      int b = B;
      if (stapel_s > 0) b = (t_gruppe > 0) ? (int)(stapel_s / t_gruppe) : 1;
      return std::max(1, std::min(std::min(b, B), rest));
   }

   bool anlegen(const int32_t *cls) {
      if (const char *cu = getenv("TS_GPU_CU")) {                // Schonbetrieb: nur n CUs, paarweise (ein WGP = 2 CUs)
         const int ncu = 64, soll = std::max(2, std::min(ncu, atoi(cu))) & ~1;
         uint32_t maske[2] = {0, 0};
         for (int w = 0; w < soll / 2; ++w) {                    // WGPs gleichmaessig verteilt
            const int wgp = (int)((long)w * (ncu / 2) / (soll / 2));
            maske[(2 * wgp) / 32] |= 3u << ((2 * wgp) % 32);
         }
         HCK(hipExtStreamCreateWithCUMask(&st, 2, maske));
      } else {
         HCK(hipStreamCreate(&st));
      }
      HCK(hipMalloc(&dL, np * TSG_L * 8 * (size_t)B)); HCK(hipMalloc(&dR, (size_t)B * TSG_W * TSG_L * 8));
      HCK(hipMalloc(&dS, tsg_skizze_bytes(n, B)));
      HCK(hipMalloc(&dFt, (size_t)B * ncomp * m * m * TSG_L * 8)); HCK(hipMalloc(&dFd, (size_t)B * ncomp * m * TSG_L * 8));
      HCK(hipMalloc(&dcls, np * 4)); HCK(hipMalloc(&dkls, (size_t)n * 4));
      HCK(hipMemcpy(dcls, cls, np * 4, hipMemcpyHostToDevice)); HCK(hipMemcpy(dkls, kls.data(), (size_t)n * 4, hipMemcpyHostToDevice));
      for (Puffer &q : p) {
         HCK(hipHostMalloc(&q.h_tab, n_tab(B) * 8)); HCK(hipHostMalloc(&q.h_cs, n_ph(B) * 8)); HCK(hipHostMalloc(&q.h_sn, n_ph(B) * 8));
         HCK(hipHostMalloc(&q.h_sv, n_sv(B) * 8)); HCK(hipHostMalloc(&q.h_vh, n_vh() * 8)); HCK(hipHostMalloc(&q.h_info, (size_t)B * TSG_L * 4));
         HCK(hipMalloc(&q.d_tab, n_tab(B) * 8)); HCK(hipMalloc(&q.d_cs, n_ph(B) * 8)); HCK(hipMalloc(&q.d_sn, n_ph(B) * 8));
         HCK(hipMalloc(&q.d_sv, n_sv(B) * 8)); HCK(hipMalloc(&q.d_vh, n_vh() * 8)); HCK(hipMalloc(&q.d_info, (size_t)B * TSG_L * 4));
         HCK(hipEventCreate(&q.start)); HCK(hipEventCreate(&q.fertig));       // mit Zeitmessung (Stapelgroesse)
      }
      return true;
   }
   void freigeben() {
      for (Puffer &q : p) {
         (void)hipHostFree(q.h_tab); (void)hipHostFree(q.h_cs); (void)hipHostFree(q.h_sn); (void)hipHostFree(q.h_sv); (void)hipHostFree(q.h_vh); (void)hipHostFree(q.h_info);
         (void)hipFree(q.d_tab); (void)hipFree(q.d_cs); (void)hipFree(q.d_sn); (void)hipFree(q.d_sv); (void)hipFree(q.d_vh); (void)hipFree(q.d_info);
         if (q.start) (void)hipEventDestroy(q.start);
         if (q.fertig) (void)hipEventDestroy(q.fertig);
      }
      (void)hipFree(dL); (void)hipFree(dR); (void)hipFree(dS); (void)hipFree(dFt); (void)hipFree(dFd); (void)hipFree(dcls); (void)hipFree(dkls);
      if (st) (void)hipStreamDestroy(st);
   }

   // CPU: Tabellen fuer den Stapel aus b Gruppen ab Frequenz f0 (0-basiert) in Puffer q
   void vorbereiten(Puffer &q, int f0, int b) {
      q.f0 = f0; q.b = b; q.nl = std::min(TSG_L * b, nfreq - f0);
      std::vector<double> tmp(2 * (size_t)K), fr(TSG_L); std::vector<int32_t> ifr(TSG_L);
      for (int g = 0; g < b; ++g) {
         const int gl = std::max(0, std::min(TSG_L, q.nl - g * TSG_L));               // gueltige Lanes dieser Gruppe
         const int basis = f0 + g * TSG_L;
         const int letzt = std::min(basis + std::max(gl, 1), nfreq) - 1;               // leere Gruppe: letzte Frequenz
         for (int l = 0; l < TSG_L; ++l) { const int f = (l < gl) ? basis + l : letzt; fr[l] = freq[f]; ifr[l] = f + 1; }
         tsgp_exp(K, distk, distuk, uhub, fr.data(), TSG_L, incdec, incohb, q.h_tab + (size_t)g * 2 * K * TSG_L, tmp.data());
         for (int c = 0; c < ncomp; ++c) {
            const size_t gc = (size_t)g * ncomp + c;
            tsgp_phasen(n, phase, ldp, nfp, ifr.data(), TSG_L, ivec + c, q.h_cs + gc * n * TSG_L, q.h_sn + gc * n * TSG_L);
            tsgp_s(m, rep.data(), S, nfs, nps, ifr.data(), TSG_L, ivec + c, q.h_sv + gc * m * TSG_L);
         }
      }
   }

   // GPU-Arbeit eines Stapels aus Puffer q auf den Strom legen
   bool folge(Puffer &q) {
      const int b = q.b;               // Gruppen dieses Stapels; Puffer und Skizze gruppenweise, die ersten b genuegen
      HCK(hipMemcpyAsync(q.d_tab, q.h_tab, n_tab(b) * 8, hipMemcpyHostToDevice, st));
      HCK(hipMemcpyAsync(q.d_cs, q.h_cs, n_ph(b) * 8, hipMemcpyHostToDevice, st));
      HCK(hipMemcpyAsync(q.d_sn, q.h_sn, n_ph(b) * 8, hipMemcpyHostToDevice, st));
      HCK(hipMemcpyAsync(q.d_sv, q.h_sv, n_sv(b) * 8, hipMemcpyHostToDevice, st));
      HCK(hipMemsetAsync(q.d_info, 0, (size_t)B * TSG_L * 4, st));
      tsg_k_gather<<<dim3((unsigned)((np + 7) / 8), (unsigned)b), 256, 0, st>>>(dL, np, dcls, q.d_tab, 2 * K);
      tsg_k_faktoren<<<dim3((unsigned)m, (unsigned)(b * ncomp)), TSG_L, 0, st>>>(q.d_sv, dFt, dFd, m, ncomp);
      tsg_chol(dL, n, b, dR, q.d_info, skip ? dS : nullptr, st);
      tsg_k_h2c<<<dim3((unsigned)n, (unsigned)b), TSG_L, 0, st>>>(dL, np, n, dkls, m, dFt, dFd, q.d_cs, q.d_sn, ncomp, q.d_vh, B);
      HCK(hipGetLastError());
      HCK(hipMemcpyAsync(q.h_vh, q.d_vh, n_vh() * 8, hipMemcpyDeviceToHost, st));
      HCK(hipMemcpyAsync(q.h_info, q.d_info, (size_t)B * TSG_L * 4, hipMemcpyDeviceToHost, st));
      return true;
   }

   // GPU: Stapel einreihen (asynchron)
   bool einreihen(Puffer &q) {
      HCK(hipEventRecord(q.start, st));
      if (!folge(q)) return false;
      HCK(hipEventRecord(q.fertig, st));
      return true;
   }

   // CPU: warten, INFO pruefen, V einsortieren. Rueckgabe: 0 ok, sonst kleinste gescheiterte Frequenz (1-basiert).
   int abschliessen(Puffer &q) {
      if (hipEventSynchronize(q.fertig) != hipSuccess) return -1;
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, q.start, q.fertig) == hipSuccess && q.b > 0) {
         t_gruppe = ms * 1e-3 / q.b; gpu_summe += ms * 1e-3; t_letzter = ms * 1e-3;
      }
      ++stapel; gruppen += q.b;
      for (int l = 0; l < q.nl; ++l)
         if (q.h_info[l] != 0) return q.f0 + l + 1;                                      // aufsteigend: erste ist kleinste
      for (int c = 0; c < ncomp; ++c)
         for (int i = 0; i < n; ++i)                                                       // V(2f-1:2f, i, c): zusammenhaengend
            memcpy(V + 2 * (size_t)q.f0 + (size_t)ldv * ((size_t)i + (size_t)npv * (size_t)(ivec - 1 + c)),
                   q.h_vh + ((size_t)c * n + i) * (size_t)(2 * TSG_L * B), sizeof(double) * 2 * (size_t)q.nl);
      return 0;
   }
};

int geraet_suchen(hipDeviceProp_t *prop) {
   int nd = 0;
   if (hipGetDeviceCount(&nd) != hipSuccess) return -1;
   for (int i = 0; i < nd; ++i) if (hipGetDeviceProperties(prop, i) == hipSuccess && strstr(prop->gcnArchName, "gfx1201")) return i;
   return -1;
}
}  // namespace

// 1, wenn TS_GPU=nur gesetzt ist: der Aufrufer darf die Frequenzschleife dann nicht auf der CPU rechnen (Fortran: Abbruch).
extern "C" int32_t tsg_pflicht(void) { return pflicht_gesetzt() ? 1 : 0; }

extern "C" void tsg_iec_lauf(const int32_t *n_, const int32_t *nfreq_, const int32_t *cls, const int32_t *k_,
                             const double *distk, const double *distuk, const double *uhub, const double *freq,
                             const double *incdec, const double *incohb,
                             const double *S, const int32_t *nfs, const int32_t *nps,
                             const double *phase, const int32_t *ldp, const int32_t *nfp,
                             double *V, const int32_t *ldv, const int32_t *npv,
                             const int32_t *ivec, const int32_t *ncomp, int32_t *fehl_ifreq, int32_t *ok)
{
   *ok = 0; *fehl_ifreq = 0;
   const char *e = getenv("TS_GPU");
   if (e && e[0] == '0') return;
   const bool pflicht = pflicht_gesetzt();
   const bool info = getenv("TS_GPU_INFO") != nullptr || pflicht;   // bei Pflicht den Ablehnungsgrund immer melden
   const char *wohin = pflicht ? "abgelehnt (TS_GPU=nur: CPU-Pfad gesperrt)" : "CPU-Pfad";
   // HSA_OVERRIDE_GFX_VERSION (z. B. 12.0.0 aus der .bashrc fuer LLM/ComfyUI) gibt die R9700 als gfx1200 aus; das
   // Code-Objekt hier ist gfx1201. HIP liest die Variable erst beim ersten API-Aufruf, also hier noch rechtzeitig.
   if (const char *ov = getenv("HSA_OVERRIDE_GFX_VERSION")) {
      if (info) fprintf(stderr, "TurbSim-GPU: HSA_OVERRIDE_GFX_VERSION=%s fuer diesen Lauf ignoriert (Code-Objekt gfx1201)\n", ov);
      unsetenv("HSA_OVERRIDE_GFX_VERSION");
   }
   hipDeviceProp_t prop;
   const int dev = geraet_suchen(&prop);
   if (dev < 0 || hipSetDevice(dev) != hipSuccess) {
      if (info) {
         const char *sicht = getenv("HIP_VISIBLE_DEVICES") ? getenv("HIP_VISIBLE_DEVICES") : getenv("ROCR_VISIBLE_DEVICES");
         fprintf(stderr, "TurbSim-GPU: kein gfx1201%s%s%s -> %s\n", sicht ? " (sichtbar nur: " : "", sicht ? sicht : "",
                 sicht ? ")" : "", wohin);
      }
      return;
   }

   Lauf r;
   r.n = *n_; r.nfreq = *nfreq_; r.K = *k_; r.ncomp = *ncomp; r.ivec = *ivec; r.np = (size_t)r.n * (r.n + 1) / 2;
   r.distk = distk; r.distuk = distuk; r.freq = freq; r.S = S; r.phase = phase; r.uhub = *uhub; r.incdec = *incdec; r.incohb = *incohb;
   r.nfs = *nfs; r.nps = *nps; r.ldp = *ldp; r.nfp = *nfp; r.V = V; r.ldv = *ldv; r.npv = *npv;
   r.skip = !(getenv("TS_GPU_SKIP") && getenv("TS_GPU_SKIP")[0] == '0');
   // Bei TS_GPU=nur ohne Vorgabe ~2 s je Stapel: bindet die Arbeit, die nach einem Halt noch auf der GPU liegt
   r.stapel_s = std::max(0.0, env_zahl("TS_GPU_STAPEL_S", pflicht ? 2.0 : 0.0));
   r.kls.resize(r.n); r.rep.resize(TSG_MAXM);
   r.m = tsgp_klassen(r.n, S, r.nfs, r.nps, r.nfreq, r.ivec, r.ncomp, r.kls.data(), r.rep.data(), TSG_MAXM);
   if (r.m <= 0) { if (info) fprintf(stderr, "TurbSim-GPU: zu viele S-Klassen -> %s\n", wohin); return; }

   // Stapelgroesse aus dem VRAM-Budget (Reserve fuer Bildschirm/System, Deckel je Prozess)
   size_t frei = 0, ges = 0;
   if (hipMemGetInfo(&frei, &ges) != hipSuccess) { if (info) fprintf(stderr, "TurbSim-GPU: hipMemGetInfo fehlgeschlagen -> %s\n", wohin); return; }
   const double res_gb = env_zahl("TS_GPU_VRAM_RESERVE_GB", 3.0);
   const size_t res = (size_t)(res_gb * (1ULL << 30));
   const size_t b1 = ges > res ? ges - res : (size_t)0, rand = (size_t)512 << 20, b2 = frei > rand ? frei - rand : (size_t)0;
   size_t budget = b1 < b2 ? b1 : b2;
   const double deckel_gb = env_zahl("TS_GPU_VRAM_MAX_GB", 0.0);
   if (deckel_gb > 0) budget = std::min(budget, (size_t)(deckel_gb * (1ULL << 30)));
   const size_t je_gruppe = r.np * TSG_L * 8 + tsg_skizze_bytes(r.n, 1) + (size_t)r.ncomp * r.m * (r.m + 1) * TSG_L * 8
                          + 2 * (2 * (size_t)r.K * TSG_L * 8 + 2 * (size_t)r.ncomp * r.n * TSG_L * 8 + (size_t)r.ncomp * r.m * TSG_L * 8
                                 + (size_t)r.ncomp * r.n * 2 * TSG_L * 8);
   const size_t fest = r.np * 4 + (size_t)r.n * 4 + ((size_t)64 << 20);
   int bmax = budget > fest ? (int)((budget - fest) / je_gruppe) : 0;
   const char *bb = getenv("TS_GPU_B"); int bwunsch = bb ? atoi(bb) : 8;
   r.B = std::min(std::min(bmax, bwunsch), (r.nfreq + TSG_L - 1) / TSG_L);
   if (r.B < 1) { if (info) fprintf(stderr, "TurbSim-GPU: VRAM reicht nicht (je Gruppe %.2f GiB, Budget %.2f GiB) -> %s\n", je_gruppe / 1073741824.0, budget / 1073741824.0, wohin); return; }
   if (!r.anlegen(cls)) { r.freigeben(); if (info) fprintf(stderr, "TurbSim-GPU: Anlegen der Puffer fehlgeschlagen -> %s\n", wohin); return; }

   Bremse br;
   br.max = env_zahl("TS_GPU_TJ_MAX", 0.0);
   br.wieder = env_zahl("TS_GPU_TJ_WIEDER", br.max - 8.0);
   br.kmax = env_zahl("TS_GPU_KANTE_MAX", 0.0);
   br.kwieder = env_zahl("TS_GPU_KANTE_WIEDER", br.kmax - 5.0);
   br.warte_max = env_zahl("TS_GPU_TJ_WARTE_MAX_S", 900.0);
   if (br.an()) br.suchen(prop);

   auto t0 = std::chrono::steady_clock::now();
   int fehl = 0, f_naechst = 0;                                                           // erste noch nicht vorbereitete Frequenz
   auto vorb = [&](Puffer &q) { const int b = r.stapel_b(f_naechst); r.vorbereiten(q, f_naechst, b); f_naechst += TSG_L * b; };
   const double pause = std::max(0.0, std::min(0.9, env_zahl("TS_GPU_PAUSE", 0.0)));
   vorb(r.p[0]);
   if (pause > 0.0) {                                                                     // Schonbetrieb: Stapel einzeln, danach Ruhe
      for (int s = 0; !fehl; ++s) {
         Puffer &q = r.p[s & 1];
         if (!br.warten(r.t_letzter)) { fehl = -2; break; }
         const auto ta = std::chrono::steady_clock::now();
         if (!r.einreihen(q)) { fehl = -1; break; }
         fehl = r.abschliessen(q);
         const double gpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
         if (fehl || f_naechst >= r.nfreq) break;
         const auto tb = std::chrono::steady_clock::now();
         vorb(r.p[(s + 1) & 1]);
         const double rest = gpu_s * pause / (1.0 - pause) - std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count();
         if (rest > 0) std::this_thread::sleep_for(std::chrono::duration<double>(rest));
      }
   } else {
      int offen = -1;                                                                     // offen: Puffer im Flug
      for (int s = 0;; ++s) {
         Puffer &q = r.p[s & 1];
         if (offen >= 0 && br.nah()) {                                                    // nahe der Grenze: erst leerlaufen lassen,
            fehl = r.abschliessen(r.p[offen & 1]); offen = -1; if (fehl) break;          // damit die Pause sofort wirkt
         }
         if (!br.warten(r.t_letzter)) { fehl = -2; break; }
         if (!r.einreihen(q)) { fehl = -1; break; }
         if (offen >= 0) { fehl = r.abschliessen(r.p[offen & 1]); if (fehl) break; }    // vorigen Stapel einsortieren
         offen = s;
         if (f_naechst >= r.nfreq) break;
         vorb(r.p[(s + 1) & 1]);                                                          // naechsten vorbereiten, GPU rechnet
      }
      if (!fehl && offen >= 0) fehl = r.abschliessen(r.p[offen & 1]);
   }
   (void)hipStreamSynchronize(r.st);
   r.freigeben();
   if (fehl < 0) {                                                                        // V wird vom CPU-Pfad neu berechnet
      if (info) fprintf(stderr, "TurbSim-GPU: %s -> %s\n", fehl == -2 ? "Zeitlimit der Waermebremse" : "HIP-Fehler im Lauf", wohin);
      return;
   }
   if (info) {
      fprintf(stderr, "TurbSim-GPU: N = %d, %d Frequenzen, %d Komponente(n), %d S-Klassen, B = %d -> %.3f s\n", r.n, r.nfreq, r.ncomp, r.m, r.B,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      fprintf(stderr, "TurbSim-GPU: %d Stapel (%.1f Gruppen im Mittel), GPU-Zeit %.3f s, VRAM %.2f GiB", r.stapel,
              r.stapel ? (double)r.gruppen / r.stapel : 0.0, r.gpu_summe, (fest + (size_t)r.B * je_gruppe) / 1073741824.0);
      if (br.an()) fprintf(stderr, ", Waermebremse Tj %.0f/%.0f C Kante %.0f/%.0f C: %d Pausen, %.1f s gewartet, Tj hoechstens %.0f C,"
                           " Kante hoechstens %.0f C", br.max, br.wieder, br.kmax, br.kwieder, br.pausen, br.gewartet + br.ersatz_s,
                           br.tj_hoechst, br.kante_hoechst);
      if (br.lesefehler) fprintf(stderr, ", Sensor %d-mal nicht lesbar (Ersatzpausen %.1f s)", br.lesefehler, br.ersatz_s);
      fprintf(stderr, "\n");
   }
   *fehl_ifreq = fehl;
   *ok = 1;
}

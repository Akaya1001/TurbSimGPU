// t1_chol.cpp — Pruefstand Stufe G1: GPU-Cholesky (tsg_chol.hip) bitweise gegen den CPU-Kern ts_chol_spec.
//
// Kohaerenzmatrizen exakt wie im TurbSim-Bau: Punkte wie CreateGrid, Dist/DistU in Fortran-Reihenfolge, Dist-Klassen
// und exp-Tabellen ueber die echten glibc/libmvec-Funktionen (ts_kern.c: ts_cls_build, ts_exp_tab, ts_coh_fill).
// Referenz: ts_chol_spec OHNE Skip und ohne Zeilenprofil (streng bitgleich zur dpptrf-Kette inkl. Nullvorzeichen).
// Massstab: dicht alle Bits je Lane gleich; mit Auslassen (G2) alle Werte gleich (nur Vorzeichen exakter Nullen
// duerfen abweichen, wie im CPU-Kern), INFO gleich. Beide Laeufe in jedem Aufruf, mit Zeitvergleich.
//
// Bauen (aus turbsim/gpu/tests):  ./bau_t1.sh       Aufruf:  ./t1_chol <box> <ifreq0> <B> [schritt]
//   box: std | w340 | w960 | farm3600 | farm3599 | klein   (klein: 12x10-Gitter, schnell)
//   ifreq0: erste Frequenznummer; B Gruppen zu je 32 Frequenzen; schritt: Abstand der Frequenznummern (Vorgabe 1)
// NUR mit Freigabe auf der R9700 ausfuehren (Monitor an der iGPU). Kernel < 100 ms je Launch.
#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "../tsg_chol.hip"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP-Fehler %s bei %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

extern "C" {
void ts_cls_build(const int32_t *np, const int32_t *n, const double *dist, const double *distu, int32_t *cls, double *distk, double *distuk, int32_t *k);
void ts_exp_tab(const int32_t *k, const double *distk, const double *distuk, const double *uhub, const double *freq, const double *incdec, const double *incohb, double *tab);
void ts_coh_fill(const int32_t *n, const int32_t *indx0, const int32_t *cls, const double *tab, double *trh, int32_t *frow);
void ts_chol_spec(const int32_t *n, double *ap, const int32_t *frow, const int32_t *use_frow, int32_t *info);
void ts_chol_skip_setzen(int32_t an);
const char *ts_chol_isa_name(void);
}

struct Box { int ny, nz; double W, H, hub; bool nabe; double usr_y, usr_z; bool usr; double U, dt; int numfreq; double incohb; };

static void gitter(const Box &b, std::vector<double> &Y, std::vector<double> &Z) {   // wie skip_pruefung.gitter
   const double gy = b.W / (b.ny - 1), gz = b.H / (b.nz - 1), zb = b.hub + 0.5 * std::min(b.W, b.H) - gz * (b.nz - 1);
   if (b.usr) { Y.push_back(b.usr_y); Z.push_back(b.usr_z); }
   for (int iz = 0; iz < b.nz; ++iz)
      for (int iy = 0; iy < b.ny; ++iy) {
         const double y = -0.5 * b.W + gy * iy, z = zb + gz * iz;
         if (b.usr && y == b.usr_y && z == b.usr_z) continue;
         Y.push_back(y); Z.push_back(z);
      }
   if (b.nabe) { Y.push_back(0.0); Z.push_back(b.hub); }
}

int main(int argc, char **argv) {
   if (argc < 4) { fprintf(stderr, "Aufruf: t1_chol <std|w340|w960|farm3600|farm3599|klein> <ifreq0> <B> [schritt]\n"); return 1; }
   const std::string name = argv[1]; const int if0 = atoi(argv[2]), B = atoi(argv[3]), schritt = argc > 4 ? atoi(argv[4]) : 1;
   const double ic_iec = 0.12 / (8.1 * 42.0), u_farm = 10.0 * std::pow(156.0 / 150.0, 0.14);
   Box b;
   if (name == "std") b = {35, 35, 270, 270, 137.5, false, 0, 0, false, 11.0, 0.05, 43470, ic_iec};
   else if (name == "w340") b = {44, 35, 340, 270, 137.5, true, 0, 0, false, 11.0, 0.05, 43520, ic_iec};
   else if (name == "w960") b = {124, 35, 960, 270, 137.5, true, 0, 0, false, 11.0, 0.05, 44100, ic_iec};
   else if (name == "farm3600") b = {59, 61, 290, 300, 156.0, false, 12.251, 141.0, true, u_farm, 0.42, 5016, 0.00035273};
   else if (name == "farm3599") b = {59, 61, 290, 300, 156.0, false, 10.0, 141.0, true, u_farm, 0.42, 5016, 0.00035273};
   else if (name == "klein") b = {12, 10, 90, 72, 137.5, false, 0, 0, false, 11.0, 0.05, 43470, ic_iec};
   else { fprintf(stderr, "unbekannte Box %s\n", name.c_str()); return 1; }

   std::vector<double> Y, Z; gitter(b, Y, Z);
   const int n = (int)Y.size(); const size_t np = (size_t)n * (n + 1) / 2;
   // Zeitmodus (T1_ZEIT_B=<B>): eine Gruppe (32 Frequenzen) rechnen, auf der GPU B-fach vervielfachen, nur die Zeit
   // des Skip-Kerns messen (keine CPU-Referenz; die Bitgleichheit haengt nicht von B ab und ist fuer B = 1..4 belegt).
   const int zeit_B = getenv("T1_ZEIT_B") ? atoi(getenv("T1_ZEIT_B")) : 0;
   std::vector<double> dist(np), distu(np);
   { size_t e = 0; for (int j = 0; j < n; ++j) for (int i = j; i < n; ++i, ++e) {
        const double dy = Y[i] - Y[j], dz = Z[i] - Z[j]; dist[e] = std::sqrt(dy * dy + dz * dz); distu[e] = dist[e] / b.U; } }
   std::vector<int32_t> cls(np); std::vector<double> distk(np), distuk(np); int32_t K = 0, n32 = n, np32 = (int32_t)np;
   ts_cls_build(&np32, &n32, dist.data(), distu.data(), cls.data(), distk.data(), distuk.data(), &K);
   const double delf = 1.0 / (2.0 * b.numfreq * b.dt), incdec = 12.0;
   printf("Box %s: N = %d, K = %d, %d Gruppen x 32 Frequenzen ab IFreq %d (Schritt %d, f %.4f..%.4f Hz), CPU-Kern %s\n", name.c_str(), n, K, B,
          if0, schritt, if0 * delf, (if0 + (32 * B - 1) * schritt) * delf, ts_chol_isa_name());

   if (zeit_B > 0) {
      std::vector<double> hA1(np * 32), tab1(2 * (size_t)K), trh1(np); std::vector<int32_t> fr1(n);
      for (int q = 0; q < 32; ++q) {
         const double f = (if0 + q * schritt) * delf; int32_t ix0 = 1;
         ts_exp_tab(&K, distk.data(), distuk.data(), &b.U, &f, &incdec, &b.incohb, tab1.data());
         ts_coh_fill(&n32, &ix0, cls.data(), tab1.data(), trh1.data(), fr1.data());
         for (size_t e = 0; e < np; ++e) hA1[e * 32 + q] = trh1[e];
      }
      int dev = -1, nd = 0; CK(hipGetDeviceCount(&nd));
      for (int i = 0; i < nd; ++i) { hipDeviceProp_t pp; CK(hipGetDeviceProperties(&pp, i)); if (strstr(pp.gcnArchName, "gfx1201")) { dev = i; break; } }
      CK(hipSetDevice(dev));
      const size_t g1 = np * 32 * 8, bed = g1 * zeit_B + tsg_skizze_bytes(n, zeit_B);
      size_t frei, ges; CK(hipMemGetInfo(&frei, &ges));
      if (bed + (3ULL << 30) > ges || bed + (512ULL << 20) > frei) { fprintf(stderr, "VRAM-Budget: Bedarf %.2f GiB\n", bed / 1073741824.0); return 3; }
      double *dA0, *dL, *dR; int *dinfo; int16_t *dS;
      CK(hipMalloc(&dA0, g1)); CK(hipMalloc(&dL, g1 * zeit_B)); CK(hipMalloc(&dR, (size_t)zeit_B * 8 * 32 * 8)); CK(hipMalloc(&dinfo, (size_t)zeit_B * 32 * 4)); CK(hipMalloc(&dS, tsg_skizze_bytes(n, zeit_B)));
      CK(hipMemcpy(dA0, hA1.data(), g1, hipMemcpyHostToDevice));
      hipStream_t st; CK(hipStreamCreate(&st)); hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
      float best = 1e30f;
      for (int wdh = 0; wdh < 2; ++wdh) {
         for (int g = 0; g < zeit_B; ++g) CK(hipMemcpy(dL + (size_t)g * np * 32, dA0, g1, hipMemcpyDeviceToDevice));
         CK(hipMemset(dinfo, 0, (size_t)zeit_B * 32 * 4));
         CK(hipEventRecord(e0, st)); tsg_chol(dL, n, zeit_B, dR, dinfo, dS, st); CK(hipEventRecord(e1, st));
         CK(hipEventSynchronize(e1)); CK(hipGetLastError());
         float ms; CK(hipEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
      }
      printf("ZEIT %s B=%d: GPU %.1f ms fuer %d Matrizen = %.3f ms/Matrix (Skip)\n", name.c_str(), zeit_B, best, 32 * zeit_B, best / (32 * zeit_B));
      return 0;
   }
   // A je Frequenz (CPU, Produktionsbausteine), frequenzverschraenkt; CPU-Referenz je Frequenz
   const size_t tot = np * 32 * (size_t)B;
   std::vector<double> hA(tot), ref(np), tab(2 * (size_t)K), trh(np); std::vector<int32_t> frow(n), refinfo(32 * B);
   std::vector<std::vector<double>> refL(32 * B);
   ts_chol_skip_setzen(0);
   auto t0 = std::chrono::steady_clock::now();
   for (int q = 0; q < 32 * B; ++q) {
      const double f = (if0 + q * schritt) * delf; int32_t ix0 = 1;
      ts_exp_tab(&K, distk.data(), distuk.data(), &b.U, &f, &incdec, &b.incohb, tab.data());
      ts_coh_fill(&n32, &ix0, cls.data(), tab.data(), trh.data(), frow.data());
      const int g = q / 32, l = q % 32;
      // INFO-Test (T1_INFO=<spalte>): in jeder 5. Lane die Diagonale dieser Spalte (1-basiert) auf -5 setzen
      if (const char *ie = getenv("T1_INFO")) { const int k = atoi(ie) - 1; if (l % 5 == 0 && k >= 0 && k < n) trh[(size_t)k * n - (size_t)k * (k - 1) / 2] = -5.0; }
      for (size_t e = 0; e < np; ++e) hA[((size_t)g * np + e) * 32 + l] = trh[e];
      int32_t nul = 0, inf = 0; ref = trh; ts_chol_spec(&n32, ref.data(), frow.data(), &nul, &inf);
      refinfo[q] = inf; refL[q] = ref;
   }
   const double t_cpu = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

   int dev = -1, nd = 0; CK(hipGetDeviceCount(&nd));
   for (int i = 0; i < nd; ++i) { hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, i)); if (strstr(p.gcnArchName, "gfx1201")) { dev = i; break; } }
   if (dev < 0) { fprintf(stderr, "kein gfx1201\n"); return 2; }
   CK(hipSetDevice(dev));
   size_t frei, ges; CK(hipMemGetInfo(&frei, &ges));
   const size_t bedarf = tot * 8 + (size_t)B * 8 * 32 * 8 + (size_t)B * 32 * 4;
   if (bedarf + (3ULL << 30) > ges || bedarf + (512ULL << 20) > frei) { fprintf(stderr, "VRAM-Budget: Bedarf %.2f GiB, frei %.2f GiB\n", bedarf / 1073741824.0, frei / 1073741824.0); return 3; }
   double *dL, *dR; int *dinfo; int16_t *dS;
   CK(hipMalloc(&dL, tot * 8)); CK(hipMalloc(&dR, (size_t)B * 8 * 32 * 8)); CK(hipMalloc(&dinfo, (size_t)B * 32 * 4));
   CK(hipMalloc(&dS, tsg_skizze_bytes(n, B)));
   hipStream_t st; CK(hipStreamCreate(&st)); hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
   std::vector<double> hL(tot); std::vector<int> ginfo(32 * B);
   const double flop = (double)n * n * n / 3.0 * 32 * B;
   int n_info = 0; for (int q = 0; q < 32 * B; ++q) n_info += refinfo[q] != 0;
   int fehler = 0; float t_mod[2] = {0, 0};
   const char *mname[2] = {"G1 dicht", "G2 Skip "};
   for (int modus = 0; modus < 2; ++modus) {
      if (modus == 0 && getenv("T1_NUR_SKIP")) continue;
      float best = 1e30f;
      for (int wdh = 0; wdh < 2; ++wdh) {
         CK(hipMemcpy(dL, hA.data(), tot * 8, hipMemcpyHostToDevice)); CK(hipMemset(dinfo, 0, (size_t)B * 32 * 4));
         CK(hipEventRecord(e0, st)); tsg_chol(dL, n, B, dR, dinfo, modus ? dS : nullptr, st); CK(hipEventRecord(e1, st));
         CK(hipEventSynchronize(e1)); CK(hipGetLastError());
         float ms; CK(hipEventElapsedTime(&ms, e0, e1)); best = std::min(best, ms);
      }
      t_mod[modus] = best;
      CK(hipMemcpy(hL.data(), dL, tot * 8, hipMemcpyDeviceToHost)); CK(hipMemcpy(ginfo.data(), dinfo, (size_t)B * 32 * 4, hipMemcpyDeviceToHost));
      size_t bits_abw = 0, wert_abw = 0, info_abw = 0, gepr = 0; int erste = -1;
      for (int q = 0; q < 32 * B; ++q) {
         const int g = q / 32, l = q % 32;
         if (ginfo[q] != refinfo[q]) { ++info_abw; continue; }
         if (refinfo[q] != 0) continue;                            // nach INFO > 0 sind die Werte nicht definiert
         for (size_t e = 0; e < np; ++e) {
            const double a = refL[q][e], v = hL[((size_t)g * np + e) * 32 + l]; uint64_t ua, uv; memcpy(&ua, &a, 8); memcpy(&uv, &v, 8);
            ++gepr;
            if (ua != uv) { ++bits_abw; if (a != v) { ++wert_abw; if (erste < 0) { erste = q; printf("  erste Abweichung (%s): Frequenz %d, Element %zu: cpu %a gpu %a\n", mname[modus], if0 + q * schritt, e, a, v); } } }
         }
      }
      printf("%s %s: %zu Werte geprueft | Bitabweichungen %zu, davon Wertabweichungen %zu | INFO-Abweichungen %zu (Lanes mit INFO > 0: %d)\n",
             mname[modus], name.c_str(), gepr, bits_abw, wert_abw, info_abw, n_info);
      printf("%s %s: GPU %.1f ms fuer %d Matrizen = %.3f ms/Matrix = %.0f GFLOP/s (dicht gerechnet)\n",
             mname[modus], name.c_str(), best, 32 * B, best / (32 * B), flop / (best * 1e-3) / 1e9);
      if (info_abw || wert_abw || (modus == 0 && bits_abw)) fehler = 1;   // mit Skip duerfen nur Nullvorzeichen abweichen
   }
   if (t_mod[0] > 0 && t_mod[1] > 0) printf("G2 %s: Faktor dicht/Skip %.2f | CPU-Referenz %.1f s (1 Faden)\n", name.c_str(), t_mod[0] / t_mod[1], t_cpu);
   CK(hipFree(dL)); CK(hipFree(dR)); CK(hipFree(dinfo)); CK(hipFree(dS));
   return fehler;
}

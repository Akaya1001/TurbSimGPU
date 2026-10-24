// t0_gpu.cpp — Stufe-0-GPU-Messungen fuer den TurbSim-Port (R9700 / gfx1201).
//
// NUR nach Freigabe ausfuehren (die Karte treibt den Bildschirm; Monitor an die iGPU).
// Uebersetzen: hipcc -O3 --offload-arch=gfx1201 -ffp-contract=off -fno-fast-math \
//              -fno-gpu-approx-transcendentals -std=c++17 t0_gpu.cpp -o t0_gpu
// Aufruf:      ./t0_gpu info     nur Geraetewahl und VRAM-Budget, keine Rechnung
//              ./t0_gpu fma [B]  T0.1  FP64-FMA-Spitze (16 unabhaengige Ketten je Thread), B Workgroups (Vorgabe 512)
//              ./t0_gpu rund [N] T0.2  Rundung von sqrt, 1/x, a*b, a+b, fma gegen die CPU (N Werte, Vorgabe 1e8)
//              ./t0_gpu chol     T0.8  rocsolver d/spotrf_strided_batched an TurbSim-Kohaerenzmatrizen (Zeit, GFLOP/s, FP32-Fehler)
//              ./t0_gpu pcie     T0.6  H2D/D2H pinned und pageable, 16 MB und 256 MB
//              ./t0_gpu wd <s>   T0.7  Spin-Kernel von <s> Sekunden (Watchdog-Probe; danach journalctl -k)
//              ./t0_gpu dp [teil] G0    Mikrotests fuer den bitgleichen Skip-Kern (Plan 14.10.): maske | mix | sprung |
//                                       scan | dichte | band — ohne [teil] alle nacheinander
// Alle Kernel bleiben kurz (< 100 ms), ausser der ausdruecklichen Watchdog-Probe.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP-Fehler %s bei %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

// VRAM-Budget: TS_GPU_VRAM_RESERVE_GB (Vorgabe 3) bleiben fuer Bildschirm/System frei.
static size_t vram_budget() {
   size_t frei = 0, gesamt = 0; CK(hipMemGetInfo(&frei, &gesamt));
   const char *e = getenv("TS_GPU_VRAM_RESERVE_GB"); double res_gb = e ? atof(e) : 3.0;
   size_t res = (size_t)(res_gb * (1ULL << 30));
   size_t b1 = gesamt > res ? gesamt - res : 0;
   size_t b2 = frei > (512ULL << 20) ? frei - (512ULL << 20) : 0;
   size_t b = b1 < b2 ? b1 : b2;
   printf("VRAM: gesamt %.2f GiB, frei %.2f GiB, Reserve %.1f GiB -> Budget %.2f GiB\n", gesamt / 1073741824.0, frei / 1073741824.0, res_gb, b / 1073741824.0);
   return b;
}
static void *dalloc(size_t n) {   // hipMalloc mit Budgetpruefung (nie hipMallocAsync)
   static size_t verbraucht = 0, budget = 0; if (!budget) budget = vram_budget();
   if (verbraucht + n > budget) { fprintf(stderr, "VRAM-Budget ueberschritten (%zu + %zu > %zu)\n", verbraucht, n, budget); exit(3); }
   void *p; CK(hipMalloc(&p, n)); verbraucht += n; return p;
}

static int geraet_waehlen() {
   int n = 0; CK(hipGetDeviceCount(&n));
   for (int i = 0; i < n; ++i) { hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, i)); if (strstr(p.gcnArchName, "gfx1201")) { CK(hipSetDevice(i)); printf("Geraet %d: %s (%s), %d CUs, %.0f MHz\n", i, p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate / 1000.0); return i; } }
   fprintf(stderr, "kein gfx1201 gefunden\n"); exit(2);
}

// ---------------------------------------------------------------- T0.1 FP64-FMA-Spitze
__global__ void k_fma(double *out, int iters, double a, double b) {
   double x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
   double y0 = x0 * 0.5, y1 = x1 * 0.5, y2 = x2 * 0.5, y3 = x3 * 0.5, y4 = x4 * 0.5, y5 = x5 * 0.5, y6 = x6 * 0.5, y7 = x7 * 0.5;
   for (int i = 0; i < iters; ++i) {
      x0 = fma(a, x0, b); x1 = fma(a, x1, b); x2 = fma(a, x2, b); x3 = fma(a, x3, b);
      x4 = fma(a, x4, b); x5 = fma(a, x5, b); x6 = fma(a, x6, b); x7 = fma(a, x7, b);
      y0 = fma(a, y0, b); y1 = fma(a, y1, b); y2 = fma(a, y2, b); y3 = fma(a, y3, b);
      y4 = fma(a, y4, b); y5 = fma(a, y5, b); y6 = fma(a, y6, b); y7 = fma(a, y7, b);
   }
   out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7;
}
static void t_fma(int blocks) {
   const int threads = 256, iters = 2000;   // ~ 64 CU x 8 Workgroups; 2,6e9 FMA je Launch -> < 10 ms
   double *d = (double *)dalloc(sizeof(double) * blocks * threads);
   k_fma<<<blocks, threads>>>(d, 10, 0.999, 1e-3); CK(hipDeviceSynchronize());
   for (int w = 0; w < 3; ++w) {
      auto t0 = std::chrono::steady_clock::now();
      k_fma<<<blocks, threads>>>(d, iters, 0.999, 1e-3); CK(hipDeviceSynchronize());
      double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      double fma_n = (double)blocks * threads * iters * 16;
      printf("T0.1 FP64-FMA: %.3f ms, %.3f TFMA/s = %.2f TFLOP/s\n", s * 1e3, fma_n / s / 1e12, 2 * fma_n / s / 1e12);
   }
   CK(hipFree(d));
}

// ---------------------------------------------------------------- T0.2 Rundung
__global__ void k_rund(const double *a, const double *b, const double *c, double *o, size_t n) {
   size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
   const double x = a[i], y = b[i], z = c[i];
   o[5 * i + 0] = sqrt(fabs(x));
   o[5 * i + 1] = 1.0 / x;
   o[5 * i + 2] = x * y;
   o[5 * i + 3] = x + y;
   o[5 * i + 4] = fma(x, y, z);
}
static double zufall_double(std::mt19937_64 &g) {   // Exponent gleichverteilt inkl. Subnormale und +-0
   std::uniform_int_distribution<int> e(0, 2046); std::uniform_int_distribution<uint64_t> m(0, (1ULL << 52) - 1);
   int k = e(g); uint64_t bits = ((uint64_t)k << 52) | m(g); if ((g() & 255) == 0) bits = (g() & 1) ? 0x8000000000000000ULL : 0;  // gelegentlich +-0
   if (g() & 1) bits |= 0x8000000000000000ULL; double d; memcpy(&d, &bits, 8); return d;
}
static void t_rund(size_t n) {
   // blockweise: je Block 4 M Werte erzeugen, auf der GPU rechnen, gegen die CPU pruefen (Host-RAM ~260 MB)
   const size_t chunk = 1 << 22;
   std::mt19937_64 g(12345);
   std::vector<double> a(chunk), b(chunk), c(chunk), o(5 * chunk);
   double *da = (double *)dalloc(8 * chunk), *db = (double *)dalloc(8 * chunk), *dc = (double *)dalloc(8 * chunk), *dout = (double *)dalloc(8 * 5 * chunk);
   size_t bad[5] = {0, 0, 0, 0, 0}, sub = 0; const char *name[5] = {"sqrt", "1/x", "a*b", "a+b", "fma"};
   for (size_t s0 = 0; s0 < n; s0 += chunk) {
      const size_t m = std::min(chunk, n - s0);
      for (size_t i = 0; i < m; ++i) { a[i] = zufall_double(g); b[i] = zufall_double(g); c[i] = zufall_double(g); sub += std::fpclassify(a[i]) == FP_SUBNORMAL; }
      CK(hipMemcpy(da, a.data(), 8 * m, hipMemcpyHostToDevice)); CK(hipMemcpy(db, b.data(), 8 * m, hipMemcpyHostToDevice)); CK(hipMemcpy(dc, c.data(), 8 * m, hipMemcpyHostToDevice));
      k_rund<<<(unsigned)((m + 255) / 256), 256>>>(da, db, dc, dout, m); CK(hipGetLastError()); CK(hipDeviceSynchronize());
      CK(hipMemcpy(o.data(), dout, 8 * 5 * m, hipMemcpyDeviceToHost));
      for (size_t i = 0; i < m; ++i) {
         double r[5]; r[0] = std::sqrt(std::fabs(a[i])); { volatile double x = a[i]; r[1] = 1.0 / x; } { volatile double x = a[i], y = b[i]; r[2] = x * y; r[3] = x + y; } r[4] = std::fma(a[i], b[i], c[i]);
         for (int k = 0; k < 5; ++k) { uint64_t u, v; memcpy(&u, &r[k], 8); memcpy(&v, &o[5 * i + k], 8); bool nan = std::isnan(r[k]) && std::isnan(o[5 * i + k]); if (u != v && !nan) { if (bad[k] < 3) printf("  Abweichung %s: x=%a y=%a z=%a cpu=%a gpu=%a\n", name[k], a[i], b[i], c[i], r[k], o[5 * i + k]); ++bad[k]; } }
      }
   }
   for (int k = 0; k < 5; ++k) printf("T0.2 %-5s: %zu Abweichungen von %zu (davon subnormale Eingaben gesamt: %zu)\n", name[k], bad[k], n, sub);
   CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc)); CK(hipFree(dout));
}


// ---------------------------------------------------------------- T0.8 Cholesky-Durchsatz (Herstellerbibliothek)
// Kohaerenzmatrix wie TurbSim (IEC, InCDec 12, InCohB 3,5273e-4, U = 11 m/s, f = 0,5 Hz), voll, spaltenweise.
static std::vector<double> koh_matrix(int ny, int nz, double dy, double dz, bool nabe, int &n) {
   std::vector<double> y, z;
   for (int iz = 0; iz < nz; ++iz) for (int iy = 0; iy < ny; ++iy) { y.push_back(iy * dy); z.push_back(iz * dz); }
   if (nabe) { y.push_back((ny - 1) * dy / 2); z.push_back((nz - 1) * dz / 2); }
   n = (int)y.size(); std::vector<double> A((size_t)n * n);
   const double f = 0.5, U = 11.0, a = 12.0, b = 0.00035273;
   for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i) {
      double d = std::sqrt((y[i] - y[j]) * (y[i] - y[j]) + (z[i] - z[j]) * (z[i] - z[j])), x = f * (d / U), w = b * d;
      A[(size_t)j * n + i] = std::exp(-(a * std::sqrt(x * x + w * w)));
   }
   return A;
}
template <typename T> static double chol_lauf(rocblas_handle h, const std::vector<double> &A64, int n, int B, std::vector<T> &L) {
   const size_t nn = (size_t)n * n; std::vector<T> hA(nn * B);
   for (int b = 0; b < B; ++b) for (size_t k = 0; k < nn; ++k) hA[b * nn + k] = (T)A64[k];
   T *dA = (T *)dalloc(sizeof(T) * nn * B); rocblas_int *dinfo = (rocblas_int *)dalloc(sizeof(rocblas_int) * B);
   double best = 1e30;
   for (int r = 0; r < 3; ++r) {
      CK(hipMemcpy(dA, hA.data(), sizeof(T) * nn * B, hipMemcpyHostToDevice)); CK(hipDeviceSynchronize());
      auto t0 = std::chrono::steady_clock::now();
      rocblas_status st;
      if constexpr (sizeof(T) == 8) st = rocsolver_dpotrf_strided_batched(h, rocblas_fill_lower, n, (double *)dA, n, (rocblas_stride)nn, dinfo, B);
      else st = rocsolver_spotrf_strided_batched(h, rocblas_fill_lower, n, (float *)dA, n, (rocblas_stride)nn, dinfo, B);
      CK(hipDeviceSynchronize());
      double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (st != rocblas_status_success) { fprintf(stderr, "rocsolver status %d\n", (int)st); exit(4); }
      best = std::min(best, sec);
   }
   std::vector<rocblas_int> info(B); CK(hipMemcpy(info.data(), dinfo, sizeof(rocblas_int) * B, hipMemcpyDeviceToHost));
   for (int b = 0; b < B; ++b) if (info[b]) { fprintf(stderr, "info[%d] = %d\n", b, info[b]); exit(5); }
   L.resize(nn); CK(hipMemcpy(L.data(), dA, sizeof(T) * nn, hipMemcpyDeviceToHost));
   CK(hipFree(dA)); CK(hipFree(dinfo));
   return best;
}
static void t_chol() {
   rocblas_handle h; if (rocblas_create_handle(&h) != rocblas_status_success) { fprintf(stderr, "rocblas_create_handle\n"); exit(4); }
   struct Fall { const char *name; int ny, nz; double dy, dz; bool nabe; int B; };
   const Fall faelle[] = { {"std 35x35", 35, 35, 270.0 / 34, 270.0 / 34, false, 32},
                           {"Farm-High 59x61", 59, 61, 5.0, 5.0, false, 8},
                           {"W960 124x35+Nabe", 124, 35, 960.0 / 123, 270.0 / 34, true, 8} };
   for (const Fall &F : faelle) {
      int n; std::vector<double> A = koh_matrix(F.ny, F.nz, F.dy, F.dz, F.nabe, n);
      const double flop = (double)n * n * n / 3.0 * F.B;
      std::vector<double> L64; std::vector<float> L32;
      double t64 = chol_lauf<double>(h, A, n, F.B, L64);
      double t32 = chol_lauf<float>(h, A, n, F.B, L32);
      double maxrel = 0, maxabs = 0;
      for (int j = 0; j < n; ++j) for (int i = j; i < n; ++i) { size_t k = (size_t)j * n + i; double d = std::fabs((double)L32[k] - L64[k]); maxabs = std::max(maxabs, d); if (std::fabs(L64[k]) > 1e-3) maxrel = std::max(maxrel, d / std::fabs(L64[k])); }
      printf("T0.8 %-17s n=%4d B=%2d | FP64 %8.1f ms = %6.1f ms/Matrix = %5.0f GFLOP/s | FP32 %7.1f ms = %5.1f ms/Matrix = %6.0f GFLOP/s | FP32-Fehler max abs %.1e, rel (|L|>1e-3) %.1e\n",
             F.name, n, F.B, t64 * 1e3, t64 * 1e3 / F.B, flop / t64 / 1e9, t32 * 1e3, t32 * 1e3 / F.B, flop / t32 / 1e9, maxabs, maxrel);
      fflush(stdout);
   }
   rocblas_destroy_handle(h);
}

// ---------------------------------------------------------------- T0.6 PCIe
static void t_pcie() {
   for (size_t mb : {16, 256}) {
      size_t n = mb << 20; double *d = (double *)dalloc(n);
      void *pg = malloc(n); void *pn; CK(hipHostMalloc(&pn, n)); memset(pg, 1, n); memset(pn, 1, n);
      for (int pinned = 0; pinned < 2; ++pinned) {
         void *h = pinned ? pn : pg;
         for (int dir = 0; dir < 2; ++dir) {
            CK(dir ? hipMemcpy(h, d, n, hipMemcpyDeviceToHost) : hipMemcpy(d, h, n, hipMemcpyHostToDevice));
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 4; ++r) CK(dir ? hipMemcpy(h, d, n, hipMemcpyDeviceToHost) : hipMemcpy(d, h, n, hipMemcpyHostToDevice));
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 4;
            printf("T0.6 %s %s %zu MB: %.1f GB/s\n", pinned ? "pinned  " : "pageable", dir ? "D2H" : "H2D", mb, n / s / 1e9);
         }
      }
      free(pg); CK(hipHostFree(pn)); CK(hipFree(d));
   }
}

// ---------------------------------------------------------------- T0.7 Watchdog
__global__ void k_spin(long long cycles, int *out) { long long t0 = clock64(); while (clock64() - t0 < cycles) {} if (threadIdx.x == 0) out[blockIdx.x] = 1; }
static void t_wd(double sek) {
   hipDeviceProp_t p; int dev; CK(hipGetDevice(&dev)); CK(hipGetDeviceProperties(&p, dev));
   long long cyc = (long long)(sek * p.clockRate * 1000.0);   // clockRate in kHz; clock64 zaehlt Shader-Takte (naeherungsweise)
   int *d = (int *)dalloc(sizeof(int) * 64);
   printf("T0.7 Spin-Kernel %.2f s (%lld Takte) ...\n", sek, cyc); fflush(stdout);
   auto t0 = std::chrono::steady_clock::now();
   k_spin<<<64, 64>>>(cyc, d); hipError_t e = hipDeviceSynchronize();
   printf("T0.7 Ergebnis: %s nach %.2f s — jetzt 'journalctl -k | tail' auf ring/timeout pruefen\n", hipGetErrorString(e), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
   CK(hipFree(d));
}

// ---------------------------------------------------------------- G0 Mikrotests (Plan 14.10.2026)
// Fragen: (1) Spart eine teilweise Exec-Maske FP64-Takte (Skip je Lane statt je Wave)? (2) Blockiert ein FP64-Befehl
// die VALU fuer Integer-Befehle anderer Waves? (3) Was kosten Ballot + uniformer Sprung? (4) Wie schnell laeuft ein
// Skizzen-Scan (int16 laden, addieren, vergleichen, Ballot)? (5) Ab wie vielen FMAs je geladenem double ist ein Kern
// FP64- statt speichergebunden? (6) Rundet die GPU-fma im Band der Skip-Regel (e(a)+e(b) in [Es-60, Es], subnormale
// s/Produkte) bitgleich zur CPU? Alle Kernel < 100 ms.
static float ev_zeit(void (*lauf)(void *), void *arg, int wdh = 3) {
   hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
   lauf(arg); CK(hipDeviceSynchronize());                                   // Aufwaermen
   float best = 1e30f;
   for (int r = 0; r < wdh; ++r) {
      CK(hipEventRecord(a)); lauf(arg); CK(hipEventRecord(b)); CK(hipEventSynchronize(b)); CK(hipGetLastError());
      float ms; CK(hipEventElapsedTime(&ms, a, b)); best = std::min(best, ms);
   }
   CK(hipEventDestroy(a)); CK(hipEventDestroy(b)); return best;
}

// (1) FP64 mit teilweiser Exec-Maske: nur die ersten <aktiv> Lanes jeder Wave rechnen 16 unabhaengige Ketten.
__global__ void k_maske(double *out, int iters, int aktiv) {
   const int lane = threadIdx.x & (warpSize - 1);
   double x[16]; for (int k = 0; k < 16; ++k) x[k] = threadIdx.x + k;
   if (lane < aktiv)
      for (int i = 0; i < iters; ++i)
#pragma unroll
         for (int k = 0; k < 16; ++k) x[k] = fma(0.999, x[k], 1e-3);
   double s = 0; for (int k = 0; k < 16; ++k) s += x[k];
   out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
struct MaskeArg { double *d; int blocks, iters, aktiv; };
static void lauf_maske(void *v) { MaskeArg *a = (MaskeArg *)v; k_maske<<<a->blocks, 256>>>(a->d, a->iters, a->aktiv); }
static void g0_maske() {
   hipDeviceProp_t p; int dev; CK(hipGetDevice(&dev)); CK(hipGetDeviceProperties(&p, dev));
   printf("G0 warpSize = %d\n", p.warpSize);
   MaskeArg a{(double *)dalloc(sizeof(double) * 512 * 256), 512, 600, 32};
   for (int akt : {32, 16, 8, 4, 1}) {
      a.aktiv = akt; float ms = ev_zeit(lauf_maske, &a);
      printf("G0 maske: %2d aktive Lanes je Wave -> %7.3f ms (volle Wave = 1,00; Verhaeltnis zur Lanezahl zeigt, ob maskierte Lanes FP64-Takte sparen)\n", akt, ms);
   }
   CK(hipFree(a.d));
}

// (2) FP64 neben Integer: modus 0 nur FP64, 1 nur Integer, 2 Waves abwechselnd FP64/Integer (gerade/ungerade Wave),
//     3 dieselbe Wave macht beides. Ueberlappen FP64 und Integer, ist (2) ~ max(0,1), sonst ~ Summe.
__global__ void k_mix(double *out, unsigned *iout, int dp_iters, int int_iters, int modus) {
   const int wave = threadIdx.x / warpSize;
   double x[8]; for (int k = 0; k < 8; ++k) x[k] = threadIdx.x + k;
   unsigned u[8]; for (int k = 0; k < 8; ++k) u[k] = threadIdx.x * 2654435761u + k;
   const bool dp = modus == 0 || modus == 3 || (modus == 2 && (wave & 1) == 0);
   const bool in = modus == 1 || modus == 3 || (modus == 2 && (wave & 1) == 1);
   if (dp) for (int i = 0; i < dp_iters; ++i)
#pragma unroll
      for (int k = 0; k < 8; ++k) x[k] = fma(0.999, x[k], 1e-3);
   if (in) for (int i = 0; i < int_iters; ++i)
#pragma unroll
      for (int k = 0; k < 8; ++k) u[k] = (u[k] ^ (u[k] >> 3)) + 0x9E3779B9u;
   double s = 0; unsigned t = 0; for (int k = 0; k < 8; ++k) { s += x[k]; t ^= u[k]; }
   out[blockIdx.x * blockDim.x + threadIdx.x] = s; iout[blockIdx.x * blockDim.x + threadIdx.x] = t;
}
struct MixArg { double *d; unsigned *u; int blocks, dpi, ini, modus; };
static void lauf_mix(void *v) { MixArg *a = (MixArg *)v; k_mix<<<a->blocks, 256>>>(a->d, a->u, a->dpi, a->ini, a->modus); }
static void g0_mix() {
   MixArg a{(double *)dalloc(sizeof(double) * 512 * 256), (unsigned *)dalloc(sizeof(unsigned) * 512 * 256), 512, 600, 0, 0};
   // Integer-Iterationen so waehlen, dass "nur Integer" etwa so lange dauert wie "nur FP64"
   a.modus = 0; a.ini = 0; float t_dp = ev_zeit(lauf_mix, &a);
   a.modus = 1; a.dpi = 0; a.ini = 4000; float t_in = ev_zeit(lauf_mix, &a);
   a.ini = (int)(4000.0 * t_dp / t_in); a.dpi = 600;
   const char *name[4] = {"nur FP64", "nur Integer", "Waves abwechselnd", "dieselbe Wave beides"};
   float t[4];
   for (int m = 0; m < 4; ++m) {
      a.modus = m; MixArg b = a; if (m == 0) b.ini = 0; if (m == 1) b.dpi = 0;
      t[m] = ev_zeit(lauf_mix, &b);
      printf("G0 mix: %-22s %7.3f ms\n", name[m], t[m]);
   }
   printf("G0 mix: abwechselnd / max(FP64, Int) = %.2f (1 = volle Ueberlappung, 2 = keine); dieselbe Wave / Summe = %.2f\n",
          t[2] / std::max(t[0], t[1]), t[3] / (t[0] + t[1]));
   CK(hipFree(a.d)); CK(hipFree(a.u));
}

// (3) Ballot + uniformer Sprung je Iteration: modus 0 ohne, 1 Ballot und Sprung (nie genommen: alle Lanes "frei"),
//     2 Ballot und Sprung mit einer aktiven Lane (Wave fuehrt den Block aus) — je Iteration 8 FMAs im Block.
__global__ void k_sprung(double *out, const int *flag, int iters, int modus) {
   double x[8]; for (int k = 0; k < 8; ++k) x[k] = threadIdx.x + k;
   const int lane = threadIdx.x & (warpSize - 1);
   for (int i = 0; i < iters; ++i) {
      bool tun;
      if (modus == 0) tun = true;
      else { const bool p = (modus == 2 && lane == (i & (warpSize - 1))) || flag[i & 1023] != 0; tun = __ballot(p) != 0; }
      if (tun) {
#pragma unroll
         for (int k = 0; k < 8; ++k) x[k] = fma(0.999, x[k], 1e-3);
      }
   }
   double s = 0; for (int k = 0; k < 8; ++k) s += x[k];
   out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
struct SprungArg { double *d; int *flag; int blocks, iters, modus; };
static void lauf_sprung(void *v) { SprungArg *a = (SprungArg *)v; k_sprung<<<a->blocks, 256>>>(a->d, a->flag, a->iters, a->modus); }
static void g0_sprung() {
   SprungArg a{(double *)dalloc(sizeof(double) * 512 * 256), (int *)dalloc(sizeof(int) * 1024), 512, 1000, 0};
   CK(hipMemset(a.flag, 0, sizeof(int) * 1024));
   const char *name[3] = {"8 FMA je Iteration, ohne Pruefung", "Ballot+Sprung, nie genommen", "Ballot+Sprung, 1 Lane aktiv"};
   float t[3];
   for (int m = 0; m < 3; ++m) { a.modus = m; t[m] = ev_zeit(lauf_sprung, &a); printf("G0 sprung: %-36s %7.3f ms\n", name[m], t[m]); }
   printf("G0 sprung: uebersprungene Iteration kostet %.0f %% einer Iteration mit 8 FP64-FMAs; 1 aktive Lane kostet %.0f %%\n",
          100.0 * t[1] / t[0], 100.0 * t[2] / t[0]);
   CK(hipFree(a.d)); CK(hipFree(a.flag));
}

// (4) Skizzen-Scan: je Lane ein int16-Paar laden (koalesziert), addieren, gegen eine Schwelle pruefen, Ballot;
//     nur sehr selten (1 von 4096) folgt ein FP64-Block. Durchsatz in Pruefungen je Sekunde und GB/s.
__global__ void k_scan(const short *sk, size_t n, int schwelle, double *out) {
   const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x, schritt = (size_t)gridDim.x * blockDim.x;
   double x = threadIdx.x; int treffer = 0;
   for (size_t i = t; i < n; i += schritt) {
      const int v = sk[i] + sk[(i * 7) % n];
      if (__ballot(v > schwelle) != 0) { x = fma(0.999, x, 1e-3); ++treffer; }
   }
   out[t] = x + treffer;
}
struct ScanArg { const short *sk; size_t n; double *out; int blocks; };
static void lauf_scan(void *v) { ScanArg *a = (ScanArg *)v; k_scan<<<a->blocks, 256>>>(a->sk, a->n, 30000, a->out); }
static void g0_scan() {
   const size_t n = (size_t)256 << 20;                                       // 256 Mi Eintraege = 512 MiB
   std::vector<short> h(n); std::mt19937 g(7); for (size_t i = 0; i < n; ++i) h[i] = (short)((g() & 4095) == 0 ? 31000 : -500 - (int)(g() & 255));
   short *d = (short *)dalloc(sizeof(short) * n); CK(hipMemcpy(d, h.data(), sizeof(short) * n, hipMemcpyHostToDevice));
   ScanArg a{d, n, (double *)dalloc(sizeof(double) * 4096 * 256), 4096};
   float ms = ev_zeit(lauf_scan, &a);
   printf("G0 scan: %.3f ms fuer %zu Pruefungen = %.1f G Pruefungen/s (%.0f GB/s int16, 2 Ladevorgaenge je Pruefung)\n",
          ms, n, n / (ms * 1e-3) / 1e9, 2.0 * n * sizeof(short) / (ms * 1e-3) / 1e9);
   CK(hipFree(d)); CK(hipFree(a.out));
}

// (5) Rechendichte: jede Lane laedt double-Werte koalesziert (256 B je Wave und Schritt) und macht r FMAs je Wert.
__global__ void k_dichte(const double *a, size_t n, int r, double *out) {
   const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x, schritt = (size_t)gridDim.x * blockDim.x;
   double x0 = 0, x1 = 1, x2 = 2, x3 = 3;
   for (size_t i = t; i < n; i += schritt) {
      const double v = a[i];
      for (int k = 0; k < r; k += 4) { x0 = fma(v, x0, 1e-3); x1 = fma(v, x1, 1e-3); x2 = fma(v, x2, 1e-3); x3 = fma(v, x3, 1e-3); }
   }
   out[t] = x0 + x1 + x2 + x3;
}
struct DichteArg { const double *a; size_t n; int r; double *out; };
static void lauf_dichte(void *v) { DichteArg *a = (DichteArg *)v; k_dichte<<<4096, 256>>>(a->a, a->n, a->r, a->out); }
static void g0_dichte() {
   const size_t n = (size_t)64 << 20;                                        // 512 MiB
   double *d = (double *)dalloc(sizeof(double) * n); CK(hipMemset(d, 0, sizeof(double) * n));
   DichteArg a{d, n, 4, (double *)dalloc(sizeof(double) * 4096 * 256)};
   for (int r : {4, 8, 16, 32, 64}) {
      a.r = r; float ms = ev_zeit(lauf_dichte, &a);
      printf("G0 dichte: %2d FMA je geladenem double -> %7.3f ms = %5.0f GB/s, %.3f TFMA/s\n", r, ms, 8.0 * n / (ms * 1e-3) / 1e9, (double)n * r / (ms * 1e-3) / 1e12);
   }
   CK(hipFree(d)); CK(hipFree(a.out));
}

// (6) fma im Band der Skip-Regel: e(a)+e(b) in [Es-60, Es], s auch subnormal/Zweierpotenz, Produkte bis in den
//     Subnormalbereich — GPU-fma bitweise gegen CPU std::fma.
__global__ void k_band(const double *a, const double *b, const double *c, double *o, size_t n) {
   size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = fma(-a[i], b[i], c[i]);
}
static double baue(int feld, uint64_t mant, bool neg) { uint64_t u = ((uint64_t)neg << 63) | ((uint64_t)feld << 52) | mant; double d; memcpy(&d, &u, 8); return d; }
static void g0_band(size_t n) {
   std::mt19937_64 g(14102026);
   std::vector<double> a(n), b(n), c(n), o(n);
   for (size_t i = 0; i < n; ++i) {
      int fs = (int)(g() % 2047); if ((g() & 15) == 0) fs = 0;                // ~6 % subnormales s
      uint64_t ms = g() & ((1ULL << 52) - 1); if ((g() & 3) == 0) ms = 0;     // Zweierpotenzen
      const int es = std::max(fs, 1) - 1023, summe = es - (int)(g() % 61);    // e(a)+e(b) in [Es-60, Es]
      int ea = -1022 + (int)(g() % 2046); ea = std::max(std::min(ea, summe + 1022), summe - 1023); ea = std::max(-1023, std::min(ea, 1023));
      int eb = summe - ea; if (eb < -1023 || eb > 1023) { eb = std::max(-1023, std::min(eb, 1023)); }
      uint64_t ma = g() & ((1ULL << 52) - 1), mb = g() & ((1ULL << 52) - 1); if ((g() & 7) == 0) ma = (1ULL << 52) - 1; if ((g() & 7) == 0) mb = (1ULL << 52) - 1;
      a[i] = baue(ea + 1023, ma, g() & 1); b[i] = baue(eb + 1023, mb, g() & 1); c[i] = baue(fs, ms, g() & 1);
   }
   double *da = (double *)dalloc(8 * n), *db = (double *)dalloc(8 * n), *dc = (double *)dalloc(8 * n), *dout = (double *)dalloc(8 * n);
   CK(hipMemcpy(da, a.data(), 8 * n, hipMemcpyHostToDevice)); CK(hipMemcpy(db, b.data(), 8 * n, hipMemcpyHostToDevice)); CK(hipMemcpy(dc, c.data(), 8 * n, hipMemcpyHostToDevice));
   for (size_t s0 = 0; s0 < n; s0 += (1 << 24)) { size_t m = std::min((size_t)1 << 24, n - s0); k_band<<<(unsigned)((m + 255) / 256), 256>>>(da + s0, db + s0, dc + s0, dout + s0, m); CK(hipDeviceSynchronize()); }
   CK(hipMemcpy(o.data(), dout, 8 * n, hipMemcpyDeviceToHost));
   size_t bad = 0, sub = 0, nontriv = 0;
   for (size_t i = 0; i < n; ++i) {
      const double r = std::fma(-a[i], b[i], c[i]); uint64_t u, v; memcpy(&u, &r, 8); memcpy(&v, &o[i], 8);
      sub += std::fpclassify(c[i]) == FP_SUBNORMAL || std::fpclassify(r) == FP_SUBNORMAL; nontriv += r != c[i];
      if (u != v) { if (bad < 3) printf("  Abweichung: a=%a b=%a s=%a cpu=%a gpu=%a\n", a[i], b[i], c[i], r, o[i]); ++bad; }
   }
   printf("G0 band: %zu Abweichungen von %zu fma (s oder Ergebnis subnormal: %zu, Ergebnis != s: %zu)\n", bad, n, sub, nontriv);
   CK(hipFree(da)); CK(hipFree(db)); CK(hipFree(dc)); CK(hipFree(dout));
}

static void g0(const std::string &teil) {
   if (teil == "alle" || teil == "maske") g0_maske();
   if (teil == "alle" || teil == "mix") g0_mix();
   if (teil == "alle" || teil == "sprung") g0_sprung();
   if (teil == "alle" || teil == "scan") g0_scan();
   if (teil == "alle" || teil == "dichte") g0_dichte();
   if (teil == "alle" || teil == "band") g0_band(20000000);
   fflush(stdout);
}

int main(int argc, char **argv) {
   if (argc < 2) { fprintf(stderr, "Aufruf: t0_gpu info | fma [B] | rund [N] | chol | pcie | wd <s> | dp [maske|mix|sprung|scan|dichte|band]\n"); return 1; }
   geraet_waehlen();
   std::string w = argv[1];
   if (w == "info") { vram_budget(); }
   else if (w == "fma") t_fma(argc > 2 ? atoi(argv[2]) : 512);
   else if (w == "rund") t_rund(argc > 2 ? (size_t)atoll(argv[2]) : (size_t)100000000);
   else if (w == "pcie") t_pcie();
   else if (w == "chol") t_chol();
   else if (w == "dp") g0(argc > 2 ? argv[2] : "alle");
   else if (w == "wd") { double t = argc > 2 ? atof(argv[2]) : 0.1; if (t > 0.2) { fprintf(stderr, "Watchdog-Probe > 0,2 s gesperrt (Bildschirm haengt an der R9700)\n"); return 1; } t_wd(t); }
   else { fprintf(stderr, "unbekannt: %s\n", argv[1]); return 1; }
   return 0;
}

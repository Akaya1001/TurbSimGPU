# TurbSimGPU

**Bit-identical, GPU-accelerated TurbSim (OpenFAST v3.5.3).**

TurbSimGPU builds TurbSim, the stochastic inflow turbulence generator of [OpenFAST](https://github.com/OpenFAST/openfast),
so that it computes the expensive part of a wind field on an AMD GPU (HIP/ROCm) or on all CPU cores, and still writes
`.bts` files that are **bit for bit identical** to the stock TurbSim v3.5.3 binary. Not "within tolerance": the same
bytes, so existing datasets stay reproducible and every deviation is a bug, not a rounding question.

| Wind field (IEA 15 MW study) | Stock TurbSim v3.5.3 | TurbSimGPU, CPU (12 threads) | TurbSimGPU, GPU (R9700) |
|---|---|---|---|
| 35×35 points, 4 320 s (639 MB `.bts`) | ~770 s | 40 s | **22 s** |
| 35×124 points, 4 320 s (2.3 GB `.bts`) | 11 300–12 700 s | 983 s | **544 s** (with thermal brake; ~300 s without) |

Details and conditions: [docs/BENCHMARKS.md](docs/BENCHMARKS.md). How bit-identity is achieved:
[docs/DESIGN.md](docs/DESIGN.md).

---

## Contents

1. [How it works (in one minute)](#1-how-it-works-in-one-minute)
2. [Requirements](#2-requirements)
3. [Build](#3-build)
4. [Quick start: one wind field](#4-quick-start-one-wind-field)
5. [Campaigns: the GPU-only planner and runner](#5-campaigns-the-gpu-only-planner-and-runner)
6. [Verify bit-identity](#6-verify-bit-identity)
7. [Environment variables of the binary](#7-environment-variables-of-the-binary)
8. [Limitations](#8-limitations)
9. [Troubleshooting](#9-troubleshooting)
10. [Tests](#10-tests)
11. [Repository layout](#11-repository-layout)
12. [License](#12-license)

---

## 1. How it works (in one minute)

Almost all of TurbSim's run time goes into one loop: for every frequency (44 100 of them for a 72-minute box at
20 Hz) it builds an N×N spatial coherence matrix (N = number of grid points, up to 4 341 here), factorises it with
Cholesky (LAPACK `dpptrf`), scales it and sums it with random phases. TurbSimGPU keeps every floating-point operation
of that loop, in the same order and with the same rounding, and removes only work that provably changes nothing:

* **Coherence**: `exp` depends only on the bit pattern of the point distance; a box has a few thousand distinct
  patterns, so `exp` is evaluated per class with the very same glibc/libmvec functions the reference uses, including
  its mix of vector and scalar `exp` calls.
* **Cholesky**: OpenBLAS `dpptrf` is, per matrix element, a fixed chain of fused multiply-adds. Any schedule that keeps
  that chain gives the same bits, so it can be tiled, parallelised and moved to the GPU.
* **Certified no-op elimination**: an FMA whose rounded result equals its input is skipped, but only if an integer
  rule on the exponents *proves* it (`e(a) + e(b) ≤ E(s) − 56`). Coherence decays exponentially, so 96–97 % of the
  FMAs disappear on typical boxes.
* **Frequency lanes on the GPU**: each GPU wavefront computes the same matrix element for 32 neighbouring frequencies,
  one per lane; their skip patterns are nearly identical, so the skipping stays wave-uniform.
* **No transcendental functions on the GPU**: GPU `exp/sin/cos` would give different bits, so the CPU provides those
  values as tables; the GPU uses only correctly rounded `fma`, `+`, `×`, `sqrt` and `1/x`.

The build is an ordinary OpenFAST v3.5.3 build with a chain of idempotent source patches applied first. The same binary
runs on the CPU when no GPU is present or wanted.

## 2. Requirements

**For bit-identity with your reference**, build both with the same toolchain. Bit-identity was verified with:

| Item | Verified version | Why it matters |
|---|---|---|
| OS / CPU | Linux x86-64 (Ubuntu 24.04) | — |
| Fortran compiler | gfortran 13.3.0 | code generation of the unpatched parts (the build script warns on other versions) |
| C library | glibc 2.39 (libm, libmvec) | `exp`, `sin`, `cos` bits come from here |
| BLAS/LAPACK of the reference | OpenBLAS (0.3.26) | the Cholesky chain reproduced is OpenBLAS `dpptrf` (FMA-based); a reference linked against netlib BLAS rounds differently |
| OpenFAST | v3.5.3 | the patches target this release |

**Build tools**: CMake ≥ 3.16, gcc, gfortran, git, Python ≥ 3.10.

**GPU path** (optional): ROCm with `hipcc` (verified 7.2.1) and an AMD GPU of architecture **gfx1201**
(Radeon AI PRO R9700 / RX 9070 family). Other AMD architectures can be selected with `GPU_ARCH`, but are untested; the
code relies on IEEE-correct FP64 `fma/sqrt/div` and no denormal flushing, which HIP provides on AMD GPUs.

**Memory** (same as stock TurbSim, plus a little):
* RAM ≈ `24·N·(NumSteps + 2·NumFreq)` bytes — 5.2 GiB for 35×35 points × 72 min, 18 GiB for 35×124 points.
* VRAM ≈ `256·N(N+1)/2` bytes per group of 32 frequencies, up to 8 groups — 2.6 GB per group for N = 4 341
  (the runner picks fewer groups when VRAM is short).

## 3. Build

```bash
git clone https://github.com/Akaya1001/TurbSimGPU.git
cd TurbSimGPU

# A dedicated OpenFAST v3.5.3 checkout: the build patches it IN PLACE (idempotent).
git clone --branch v3.5.3 --depth 1 https://github.com/OpenFAST/openfast.git ../openfast-v3.5.3

# GPU build (CPU path included)
GPU=ON JOBS=4 ./turbsim/build_turbsim_ref353.sh ../openfast-v3.5.3 build/turbsim-gpu

# ...or CPU-only build (OpenMP, scalar/AVX2/AVX-512 kernels, no ROCm needed)
JOBS=4 ./turbsim/build_turbsim_ref353.sh ../openfast-v3.5.3 build/turbsim
```

The binary ends up in `build/turbsim-gpu/modules/turbsim/turbsim`. A first build takes a few minutes (the whole NWTC
library is compiled); later builds only relink.

What the script does:

1. Applies the patch chain to `modules/turbsim/src/TSsubs.f90` and `TurbSim.f90` (order enforced; re-running is
   harmless). See [docs/DESIGN.md §5](docs/DESIGN.md#5-the-patch-chain).
2. Compiles the C kernels (`turbsim/kern/`) with `-ffp-contract=off -fno-fast-math` and explicit ISA flags, and, with
   `GPU=ON`, the HIP engine (`turbsim/gpu/`) with `hipcc --offload-arch=$GPU_ARCH -ffp-contract=off
   -fno-gpu-approx-transcendentals`.
3. Configures OpenFAST with `-DDOUBLE_PRECISION=ON -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON` and builds the `turbsim`
   target.
4. **Checks the objects**: no FMA instructions where the reference has none, no libmvec calls in the patched
   routines. A failed check aborts the build, because it would break bit-identity.

Options (environment variables): `GPU=ON|OFF`, `GPU_ARCH` (default `gfx1201`), `JOBS` (2), `OPENMP` (`ON`),
`GIT_DESCRIBE` (`v3.5.3`: the version string written into the `.bts`/`.sum` header, identical to stock v3.5.3).

> OpenFAST embeds absolute source paths in a few runtime error messages. If you want even *error output* to match a
> particular reference binary byte for byte, build from a worktree at the same absolute path. Wind-field data are
> bit-identical regardless.

## 4. Quick start: one wind field

```bash
cd examples/std120
TS_GPU=nur TS_GPU_INFO=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    ../../build/turbsim-gpu/modules/turbsim/turbsim WS_11_TI_12_Seed_1.inp
```

`TS_GPU=nur` ("GPU only") makes the frequency loop run on the GPU or not at all: if the GPU cannot do it, TurbSim
stops with an error instead of silently computing on the CPU. With `TS_GPU_INFO=1` the engine reports what it did:

```
TurbSim-GPU: N = 1225, 1449 Frequenzen, 1 Komponente(n), 1 S-Klassen, B = 8 -> 0.680 s
TurbSim-GPU: 8 Stapel (5.8 Gruppen im Mittel), GPU-Zeit 0.676 s, VRAM 1.63 GiB
```

(N points, frequencies, components per GPU call, classes of points with identical spectra, B = groups of 32
frequencies per batch; then batches, GPU time, VRAM.) The output is the usual `WS_11_TI_12_Seed_1.bts` and `.sum`.

Check it against the reference:

```bash
python3 ../../turbsim/bts_sha256.py WS_11_TI_12_Seed_1.bts
grep std120 ../REFERENCE_SHA256.txt
```

Both lines must show `5527e89371d99f3e8b594025d011d65cfdb00501cfd15ad9089d4f059529ef11`.

Without `TS_GPU=nur` the binary uses the GPU when it can and falls back to the (equally bit-identical) CPU path
otherwise; `TS_GPU=0` forces the CPU path.

## 5. Campaigns: the GPU-only planner and runner

`turbsim/gpu_windfelder.py` ("GPU wind fields") generates many boxes on the GPU without overloading the machine. It
needs only the Python standard library. Its console output is in German; a glossary is at the end of this section.

### 5.1 Plan first

```bash
python3 turbsim/gpu_windfelder.py examples --ziel out --nur-planen
```

`--ziel` is the output directory (the folder structure of the inputs is mirrored into it); `--nur-planen` means
"plan only". The script

1. **detects the system**: free RAM (`MemAvailable` minus a reserve), the GPU's VRAM (minus reserve and what others
   use), free disk, the GPU power limit and temperatures, and other processes on the GPU;
2. **measures every box with TurbSim itself** (`TS_PLAN=1`: TurbSim builds the grid, prints the sizes and exits after
   ~10 ms, no memory): points, time steps, frequencies, output grid, coherence models, output files;
3. **plans**: RAM, VRAM and `.bts` size per box (the `.bts` size is exact to the byte), how many boxes fit at the same
   time, how many fit on the disk, and an estimated duration.

Example output (paths shortened; this machine was busy with another GPU job, so the 35×124 box did not fit into the
remaining RAM):

```
System
  RAM      30,5 GiB gesamt, 17,9 GiB verfuegbar, Reserve 4 GiB -> 13,9 GiB fuer TurbSim; Swap 8,0 GiB frei (wird je Box gesperrt)
  GPU      0000:03:00.0: VRAM 31,9 GiB, belegt 0,6 GiB, Reserve 3 GiB -> 28,3 GiB fuer TurbSim
           Leistungsgrenze 210 W, Tj 39 C, Kante 36 C, 25 W, Luefter 1188 rpm
  CPU      AMD Ryzen 9 9900X3D 12-Core Processor, 24 Faeden; je Box 2 Faeden nur fuer Einlesen/FFT/Ausgabe, Tctl 83 C
  Platte   /home/user/TurbSimGPU/out: 498,3 GiB frei, Reserve 10 GiB
  Profil   sicher: Waermebremse Kante 58/53 C und Tj 92/84 C, Stapel ~6 s; Waechter beendet die Boxen bei Tj 93 C; Leistungsgrenze hoechstens 215 W

Boxen (7; Ausgabe gespiegelt nach /home/user/TurbSimGPU/out)
  Box                                      Punkte  Freq.       RAM      VRAM (B)      .bts gleichz.     GPU-Zeit  Zustand
  farm_high/HighT1.inp                       3600    374   1,0 GiB  14,1 GiB (8)   0,0 GiB       12     6 s grob  zu rechnen
  farm_low/Low.inp                           2413     42   0,5 GiB   2,1 GiB (2)   0,0 GiB       21 1 s gemessen  zu rechnen
  std120/WS_11_TI_12_Seed_1.inp              1225   1449   0,6 GiB   2,2 GiB (8)   0,0 GiB       24 1 s gemessen  zu rechnen
  std_full/WS_9_TI_12_Seed_1.inp             1225  43520   5,2 GiB   2,2 GiB (8)   0,6 GiB        2    18 s grob  zu rechnen
  w340_120/WS_11_TI_12_Seed_1_W340.inp       1541   1512   0,7 GiB   3,1 GiB (8)   0,0 GiB       22 1 s gemessen  zu rechnen
  w960_120/WS_11_TI_12_Seed_1_W960.inp       4341   2079   1,7 GiB  20,1 GiB (8)   0,1 GiB        9    11 s grob  zu rechnen
  w960_full/WS_11_TI_12_Seed_1_W960.inp      4341  44100  17,8 GiB   passt nicht   2,1 GiB        0 5 min gemessen  zu rechnen

Plan
  zu rechnen: 7 Boxen, davon machbar 6
  passt NICHT in den RAM: WS_11_TI_12_Seed_1_W960 braucht 17,8 GiB, nutzbar 13,9 GiB
  gleichzeitig: 2 (RAM/VRAM tragen 6 der kleinsten Boxen zugleich, Obergrenze --parallel 2)
  Platte: 6 von 6 Boxen passen (0,8 GiB von 488,3 GiB nutzbar)
  Dauer: ~38 s ohne Bremspausen (GPU 38 s, CPU-Anteil 11 s; 'grob' = +-50 %, wird nach jeder Box genauer)
```

Columns: points, frequencies, RAM per box, VRAM per box with the number of 32-frequency groups per batch in brackets,
`.bts` size, how many boxes of this kind fit at the same time ("gleichz."), estimated GPU time ("grob" = rough
estimate from calibration data, "gemessen" = measured on this machine in an earlier run), state.

### 5.2 Run

Drop `--nur-planen`:

```bash
python3 turbsim/gpu_windfelder.py examples --ziel out
```

Inputs can be `.inp` files or directories (searched recursively; `_templates` folders are skipped; files that are not
TurbSim inputs are ignored). Referenced user files (`UserFile`, `ProfileFile`, e.g. TIMESR time series) are copied
with the box. While running, the script prints one line per started and finished box:

```
  start WS_11_TI_12_Seed_1_W960.inp: B=8, VRAM 20,1 GiB, RAM 18,0 GiB (1 laufen)
  [1/1] WS_11_TI_12_Seed_1_W960.inp fertig in 9 min (GPU 4 min, Tj max 82 C, Kante max 58 C, 105 W im Mittel, gebremst 5 min, RSS 17,3 GiB)
```

What it does for every box:

* runs the binary with `TS_GPU=nur`: the frequency loop is computed on the GPU or the box fails, never on the CPU.
  The CPU only reads the input, draws the random phases, runs the inverse FFT and writes the file, with
  `--cpu-faeden` threads (default 2, no AVX-512);
* passes VRAM and batch limits so that parallel boxes never over-subscribe the GPU;
* starts each box inside `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0`: if the memory estimate is
  ever too small, only that box is killed, not your desktop;
* considers a box **done** only if the `.bts` has exactly the planned size, the log proves the GPU loop ran, and then
  writes a receipt `<name>.gpu.ok` with a SHA-256 of the input (and user files). On the next run, boxes with a valid
  receipt are skipped; if you edit an input, its box is recomputed. Half-written outputs are deleted.

Output files per box in the mirrored directory: `<name>.bts`, `<name>.sum`, `<name>.gpu.log` (TurbSim's output plus
the engine report), `<name>.gpu.ok`. Per run: `sensoren_<time>.csv` (GPU/CPU sensors every second, `fsync`ed so the
last lines survive a crash) and `bericht_<time>.txt` (report).

### 5.3 Protection against overheating and overload

Long full-load GPU runs can hang a whole PC (on the development machine: twice at a GPU junction temperature of
93–94 °C at 255–300 W). The runner therefore combines:

| Layer | What it does |
|---|---|
| Power limit check | refuses to start if the GPU power limit is above the profile's maximum (`sicher`: 215 W). Set the limit with your GPU tool (e.g. LACT, `rocm-smi`). |
| Thermal brake (in the binary) | before each batch: if the **edge** temperature is at or above the limit, wait until it is 5 °C lower. The hotspot follows power within a second and cannot be capped by pausing; the slow edge temperature can. Batches are kept to ~6 s. If no sensor can be read, the engine pauses as long as it computed (50 % duty) instead of running unbraked. |
| Watchdog (in the runner) | reads the sensors every 0.5 s; if the hotspot is at or above the emergency limit twice in a row, it **kills** the boxes (the GPU stops within a second; pausing the process would leave queued GPU work running), waits until the card has cooled down and retries the box once. After 3 emergency stops the run halts. |
| RAM | per-box `MemoryMax`; an emergency brake kills the newest box if free RAM drops below `--ram-not-gb`. |
| Other GPU users | refuses to start if other processes use the GPU; during a run, no new box starts while they are present. |
| Faults | halts after a GPU error or two failed boxes in a row; only one runner at a time (lock file); closing the terminal kills the boxes too. |

Profiles (`--profil`):

| Profile | Brake: edge pause / resume | Emergency (hotspot) | Max. power limit |
|---|---|---|---|
| `sicher` (default) | 58 / 53 °C | 93 °C | 215 W |
| `normal` | 62 / 57 °C | 95 °C | 255 W |
| `voll` | 66 / 61 °C | 97 °C | — |

With `sicher` and a 210 W limit, a 30-minute endurance test stayed at ≤ 85 °C hotspot without a single emergency stop.
How fast the runner can be depends mostly on your **fan curve**: with a quiet default curve the card can only shed
~80–100 W on average and the GPU computes about a third to half of the time. A steeper fan curve (e.g. 80–100 % fan
from ~55 °C edge) raises the throughput directly.

### 5.4 Endurance test

Before a long campaign on new hardware, run the protection under load:

```bash
python3 turbsim/gpu_windfelder.py examples/w960_full --ziel out --dauertest 30
```

The inputs are computed repeatedly for 30 minutes in a fresh folder `out/dauertest_<time>/` (outputs there are
deleted after each successful run; existing files elsewhere are never touched). The run stops at the first failure.
Good result: no `WAECHTER` lines, and in the `fertig` lines a hotspot ("Tj max") that stays well below the emergency
limit.

### 5.5 Options

| Option | Default | Meaning |
|---|---|---|
| `eingaben` | — | `.inp` files or directories |
| `--ziel DIR` | `./windfelder_gpu` | output directory (input structure mirrored) |
| `--nur-planen` | off | detect and plan only |
| `--profil` | `sicher` | thermal profile, see 5.3 |
| `--kante C` / `--notfall C` | profile | override the edge brake limit / the emergency hotspot limit |
| `--parallel N` | 2 | boxes at the same time (the GPU is the bottleneck; a second box only overlaps the CPU phases of the first) |
| `--ram-reserve-gb` | 4 | RAM kept free for the system |
| `--vram-reserve-gb` | 3 | VRAM kept free |
| `--platte-reserve-gb` | 10 | disk space kept free |
| `--ram-not-gb` | 1.5 | RAM emergency brake threshold |
| `--cpu-faeden N` | 2 | OpenMP threads per box for the CPU phases |
| `--binary PATH` | `build/turbsim-gpu/...` | TurbSim binary (also `TURBSIM_GPU_BIN`) |
| `--neu` | off | recompute boxes that are already done |
| `--trotz-warnung` | off | start despite warnings (other GPU users, running simulations, power limit too high or unreadable, no temperature sensor, memory cap not effective) |
| `--dauertest MIN` | — | endurance test, see 5.4 |

Shell variables that would hide or misdescribe the GPU (`HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`,
`ROCR_VISIBLE_DEVICES`, ...) are removed for the TurbSim processes automatically (the runner prints a "Hinweis" line).

### 5.6 Glossary of the German console output

| German | English |
|---|---|
| Box, Boxen | wind field(s) |
| zu rechnen / fertig / gescheitert / übersprungen / abgebrochen | to do / done / failed / skipped / aborted |
| passt nicht | does not fit (RAM, VRAM or disk) |
| gleichzeitig | at the same time |
| Wärmebremse (Waermebremse), gebremst | thermal brake, time spent braking |
| Wächter (WAECHTER), Notabschaltung | watchdog, emergency stop |
| Kante / Tj (Sperrschicht) | edge temperature / junction (hotspot) temperature |
| Leistungsgrenze | GPU power limit |
| Stapel, Gruppen | batches, groups of 32 frequencies |
| Platte | disk |
| Hinweis / WARNUNG / FEHLER | note / warning / error |

## 6. Verify bit-identity

**Against the published hashes** (no reference binary needed): `examples/REFERENCE_SHA256.txt` lists the SHA-256 of the
`.bts` data block for every example, produced by stock TurbSim v3.5.3 (see the file header for the exact environment).

```bash
python3 turbsim/bts_sha256.py out/std120/WS_11_TI_12_Seed_1.bts
```

**Against your own reference binary** (stock v3.5.3 built with OpenBLAS and the same toolchain):

```bash
python3 turbsim/vergleich_bts.py reference.bts new.bts
# BITGLEICH: ... (N data bytes, ...)   = bit-identical
```

The `.bts` header contains the generation date and time; `vergleich_bts.py` compares the header field by field
(description up to the timestamp) and the data block byte by byte.

**Stricter, before quantisation**: the `.bts` stores int16 values; ulp differences in the double-precision Fourier
coefficients could hide behind that rounding. The patched binary can dump them:

```bash
TS_DUMP_V=V.bin ./turbsim box.inp
python3 turbsim/vergleich_v.py reference_V.bin V.bin
```

For a reference dump, apply only `turbsim/patch_turbsim_dump.py` to a clean v3.5.3 tree and build it the stock way
(the patch adds the dump and nothing else). The full chain was verified this way on all example types.

## 7. Environment variables of the binary

| Variable | Default | Effect |
|---|---|---|
| `TS_GPU` | unset | unset/`1`: use the GPU if possible, else CPU; `0`: CPU only; `nur`: GPU only, abort instead of CPU fallback |
| `TS_GPU_INFO` | unset | print the engine report (always on with `TS_GPU=nur`) |
| `TS_GPU_B` | 8 | max. groups of 32 frequencies per batch (further limited by VRAM) |
| `TS_GPU_VRAM_RESERVE_GB` | 3 | VRAM left free |
| `TS_GPU_VRAM_MAX_GB` | unset | VRAM cap for this process |
| `TS_GPU_STAPEL_S` | 0 (2 with `nur`) | target GPU seconds per batch; 0 = always B groups |
| `TS_GPU_KANTE_MAX` / `TS_GPU_KANTE_WIEDER` | off / max − 5 | thermal brake on the edge temperature (°C) |
| `TS_GPU_TJ_MAX` / `TS_GPU_TJ_WIEDER` | off / max − 8 | additional check on the junction temperature (°C) |
| `TS_GPU_TJ_WARTE_MAX_S` | 900 | longest single brake pause before giving up |
| `TS_GPU_PAUSE` | 0 | fixed duty cycle: rest this fraction of the time (0–0.9) |
| `TS_GPU_CU` | all | use only n compute units (lower power, slower) |
| `TS_GPU_SKIP` | 1 | `0`: dense GPU Cholesky without skipping (testing) |
| `TS_PLAN` | unset | `1`: print sizes after the grid is built and exit (used by the runner) |
| `TS_DUMP_V` | unset | file name: write the double-precision V before the IFFT |
| `TS_ZEITEN` | unset | print wall-clock times per program phase |
| `TS_CHOL_ISA` | auto | CPU kernel: `scalar`, `avx2`, `avx512` |
| `TS_CHOL_SKIP` | 1 | `0`: CPU Cholesky without skipping |
| `OMP_NUM_THREADS` | all | CPU threads (frequency loop on the CPU path, FFT) |

None of these changes a single bit of the result; they only change where, when and how fast it is computed.

## 8. Limitations

* **Coherence model**: only the IEC model runs on the GPU. Components with `GENERAL` or `API` coherence use the CPU
  path (with `TS_GPU=nur` they are refused; the runner reports this during planning). `NONE` (e.g. v/w with IECKAI)
  is cheap and handled exactly.
* **User time series**: at most one user point (TIMESR boxes from the FAST.Farm toolbox have one) on the GPU path.
* At most 256 distinct spectra (point classes, typically one per height).
* Verified on one GPU model (gfx1201) and one software stack; see section 2.
* The build patches the OpenFAST tree in place; use a dedicated clone.
* Comments in the source and the runner's console output are in German.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TurbSim-GPU: kein gfx1201` ("no gfx1201") | `HSA_OVERRIDE_GFX_VERSION` (often set for ML tools) makes the card report another architecture; the binary ignores it before the first HIP call and the runner removes it. Also check `HIP_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`, membership in the `render`/`video` groups, and `rocminfo`. |
| `VRAM reicht nicht` ("not enough VRAM") | another process holds VRAM; lower `TS_GPU_B` or `TS_GPU_VRAM_RESERVE_GB`, or stop the other job. |
| Runner: `andere Prozesse auf der GPU` | another GPU job is running (e.g. ML training). Wait, or use `--trotz-warnung` if you know VRAM and power suffice. |
| Runner: `Leistungsgrenze ... > 215 W` | lower the power limit or choose `--profil normal`. |
| Many `gebremst` minutes | the card cannot get rid of the heat: steeper fan curve, better case airflow, or accept the slower rate. |
| `WAECHTER ... Notabschaltung` | the hotspot reached the emergency limit despite the brake: lower the power limit or the edge limit (`--kante`). |
| `passt NICHT in den RAM` | free memory (`MemAvailable`) is too small for this box after the reserve; close programs or lower `--ram-reserve-gb`. |
| Build: `FMA instructions in ...` / `libmvec calls ...` | your compiler flags or compiler version changed the generated code; use gcc/gfortran 13 and do not add `-march=native` or `-ffast-math`. |

## 10. Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
git clone --branch v3.5.3 --depth 1 https://github.com/OpenFAST/openfast.git ../openfast-v3.5.3-clean   # stays unpatched
OPENFAST_REF=../openfast-v3.5.3-clean .venv/bin/python -m pytest -q tests/
```

* `test_turbsim_kern.py`: C kernels against the real glibc/libmvec and OpenBLAS `dpptrf` (via ctypes), bit for bit.
* `test_turbsim_skip.py`: the no-op rule (boundary probes, random triples, counterexamples), reference chains, all ISAs.
* `test_turbsim_patches.py`: the patch chain on a clean v3.5.3 tree (`OPENFAST_REF`, an *unpatched* clone; skipped
  if absent): anchors found exactly once, idempotent, order, GPU code inside `#ifdef`.
* `test_turbsim_gpu_windfelder.py`: size models against measured values, planning logic, scheduling and watchdog
  with a fake TurbSim (no GPU load), `TS_PLAN` with the real binary (skipped if not built).

GPU kernel test benches for development: `turbsim/gpu/tests/` (`bau_t1.sh` builds `t1_chol`, which checks the GPU
Cholesky against the CPU specification).

## 11. Repository layout

```
turbsim/
  build_turbsim_ref353.sh   build script (patches + kernels + CMake + object checks)
  patch_turbsim_*.py        idempotent source patches for OpenFAST v3.5.3; ofpatch.py: helpers
  kern/                     C kernels: coherence tables, dpptrf chain (scalar/AVX2/AVX-512), skip rule, H2Coeffs, GPU prep
  gpu/                      HIP engine: tsg_chol.hip (Cholesky kernels), tsg_host.hip (batching, brake, interface)
  gpu/tests/                GPU test benches (t0_gpu: micro tests, t1_chol: Cholesky vs. CPU spec)
  gpu_windfelder.py         GPU-only planner and runner
  bts_sha256.py             SHA-256 of a .bts data block
  vergleich_bts.py          compare two .bts files (bit-identity)
  vergleich_v.py            compare two V dumps
  skip_pruefung.py          offline audit of the no-op rule on synthetic boxes
examples/                   TurbSim inputs (short and full-length) and REFERENCE_SHA256.txt
docs/                       DESIGN.md (method), BENCHMARKS.md (measurements)
tests/                      pytest suite
```

## 12. License

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). TurbSim and OpenFAST are © National Renewable Energy
Laboratory and also Apache-2.0; this repository contains no OpenFAST source code, only patches applied at build time.

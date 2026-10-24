# Benchmarks and verification results

All numbers were measured on one workstation in October 2026. Wall-clock times are for a single box unless stated
otherwise.

| Component | Version |
|---|---|
| CPU | AMD Ryzen 9 9900X3D, 12 cores / 24 threads |
| RAM | 32 GB (30.5 GiB usable) |
| GPU | AMD Radeon AI PRO R9700, 32 GB, gfx1201 (RDNA4); power limit 255 W, later 210 W |
| OS / toolchain | Ubuntu 24.04, gfortran 13.3.0, glibc 2.39, OpenBLAS 0.3.26, ROCm 7.2.1 |
| Reference | stock TurbSim from OpenFAST v3.5.3 (Release, `DOUBLE_PRECISION=ON`, OpenBLAS `dpptrf`) |

## Test boxes

All examples are under `examples/` (IEA 15 MW floating turbine study; IEC Kaimal turbulence, 20 Hz unless noted).

| Box | Grid | Points N | Frequencies | `.bts` size |
|---|---|---|---|---|
| `std120` | 35×35, 270 m × 270 m, 120 s | 1 225 | 1 449 | 21 MB |
| `w340_120` | 35×44, 340 m wide, 120 s | 1 541 | 1 512 | 28 MB |
| `w960_120` | 35×124 + hub point, 960 m wide, 120 s | 4 341 | 2 079 | 108 MB |
| `farm_low` | 127×19 (FAST.Farm low-res), u/v/w IEC | 2 413 | 42 | 1.2 MB |
| `farm_high` | 59×61 + 1 user point (TIMESR), u/v/w IEC | 3 600 | 374 | 16 MB |
| `std_full` | 35×35, 4 320 s | 1 225 | 43 520 | 639 MB |
| `w960_full` | 35×124 + hub point, 4 320 s | 4 341 | 44 100 | 2.3 GB |

## Full-length boxes

| Box | Stock v3.5.3 | CPU build, 12 threads | GPU build |
|---|---|---|---|
| `std_full` | ~2 550 s per box with 6 boxes in parallel (~770 s serial, estimated) | 40.4 s | **22.2 s** (GPU loop 16.4 s; 255 W, earlier kernel) |
| `w960_full` | 11 300–12 700 s | 983 s | **544 s** with the thermal brake at 210 W (GPU compute 241.5 s, brake pauses 284 s); ~300 s without pauses (estimated) |

Both full boxes are bit-identical to the stock-produced `.bts` (data block compared byte by byte; hashes in
`examples/REFERENCE_SHA256.txt`). The CPU-build numbers predate the CPU skip kernels.

## Short boxes

GPU build (kernel of 2026-10-15, 255 W, 4 OpenMP threads for the CPU part) against the stock binary and the CPU build.
The stock times were measured while the machine was busy with other jobs and are upper bounds.

| Box | Stock v3.5.3 | CPU build, 12 threads | GPU build total | of which GPU loop |
|---|---|---|---|---|
| `std120` | 26.7 s | 1.44 s | 1.27 s | 0.60 s |
| `w340_120` | 46.9 s | 2.71 s | — | 0.96 s |
| `farm_low` | 10.3 s | 0.73 s | — | 0.65 s |
| `farm_high` | 402 s | 11–14 s | 6.85 s | 6.2 s |
| `w960_120` | 794 s | 49.2 s | 12.9 s | 11.2 s |

All rows: V dump (double precision, before the IFFT) and `.bts` bit-identical to the stock binary.

## GPU Cholesky kernel

Per matrix, one 32-frequency group (`turbsim/gpu/tests/t1_chol`), values and INFO identical to the CPU specification:

| Case | With certified skipping | Dense (no skipping) |
|---|---|---|
| N = 4 341, 1 Hz | 34.2 ms | 60.4 ms |
| N = 4 341, 3 Hz | 7.3 ms | 60.4 ms |
| N = 3 600 (farm high), 0.6 Hz | 18.7 ms | 37 ms |
| N = 1 225, 1 Hz | 0.80 ms | — |

The R9700 reaches 0.36–0.41 TFMA/s in FP64 (measured); the dense kernel runs at 360–405 GFLOP/s. The GPU is fast here
because certified skipping removes 96–97 % of the FMAs on typical boxes, not because of FP64 throughput.

Remaining FMA work, averaged over frequency (fraction of the dense work):

| Box | Element-ideal | GPU tile/lane granularity |
|---|---|---|
| 35×35, U = 11 m/s | 0.043 | ≈ 0.045 |
| 35×124, U = 11 m/s | 0.034 | ≈ 0.035 |
| FAST.Farm high-res (f ≤ 1.19 Hz) | 0.24 | ≈ 0.25 |

The fraction depends on κ ∝ f/U: at U = 25 m/s about twice as much work remains as at 11 m/s. Below ~0.02–0.09 Hz
(< 1 % of the frequencies) nothing can be skipped.

## Endurance tests (GPU-only runner)

Two 30-minute tests with `w960_full` repeated (`--dauertest 30`):

| | Test 1 | Test 2 |
|---|---|---|
| Power limit | 255 W | 210 W |
| Brake | hotspot ≥ 78 °C (old design) | edge ≥ 58 °C, resume at 53 °C |
| Watchdog | `SIGSTOP` at hotspot 86 °C | `SIGKILL` at hotspot ≥ 93 °C (two readings) |
| Result | stopped manually after 16 min: hotspot peaks 95–96 °C | 32 min, 2 boxes, **no emergency stop** |
| Hotspot / edge max | 96 °C / 68 °C | 85 °C / 58 °C |
| Time per box | 8 min | 14–18 min (fan stayed at ~1 250 rpm, GPU active ~30 %) |

Lessons (see `docs/DESIGN.md` §6): the hotspot follows the power within a second and cannot be capped by pausing;
the edge temperature can. A follow-up single run of `w960_full` with the final settings (6-s batches) took 544 s and
was bit-identical. A steeper fan curve is the main lever for throughput.

## Memory model

| Box | Model (runner) | Measured peak RSS |
|---|---|---|
| `std_full` | 5.18 GiB | 4.94 GiB |
| `w960_full` | 18.0 GiB | 17.3 GiB |

The `.bts` size predicted from `TS_PLAN` matches the stock files to the byte.

## How bit-identity was verified

* `.bts`: data block compared byte by byte (`turbsim/vergleich_bts.py`) or by SHA-256 of the data block
  (`turbsim/bts_sha256.py`).
* V dump: the double-precision Fourier coefficients before the IFFT (`TS_DUMP_V`, `turbsim/vergleich_v.py`); this
  catches ulp differences that the int16 quantisation of the `.bts` would hide.
* Invariance: OpenMP threads 1/4/12, ISA scalar/AVX2/AVX-512, skip on/off, batch size b = 1…8, two processes on one
  GPU, thermal pauses: all bit-identical.
* Error path: forced non-positive-definite matrices give the reference error text byte for byte and the smallest
  failing frequency.

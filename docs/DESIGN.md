# Design: how TurbSimGPU stays bit-identical

This document explains why the accelerated TurbSim produces exactly the same bits as stock TurbSim v3.5.3, and which
parts of the computation were changed to make it fast. The short version: every floating-point operation that
determines the output is executed with the same operands, in the same order, with the same rounding. Only work whose
result is *provably* unchanged is skipped, and only scheduling, data layout and the choice of hardware differ.

Code comments and console messages in the sources are in German; this document is the English reference.

## 1. What the reference computes

The cost of TurbSim lies in `CalcFourierCoeffs_IEC` (`modules/turbsim/src/TSsubs.f90`). For every frequency `f`
(a full 35×124-point box at 20 Hz for 4 320 s has 44 100 frequencies) and every wind component with IEC coherence:

1. **Coherence matrix** (packed lower triangle, `N(N+1)/2` entries, `N` = number of points):
   `Coh_ij = exp(-InCDec · sqrt((f·r_ij/U_hub)² + (InCohB·r_ij)²))`.
2. **Cholesky factorisation** `Coh = L·Lᵀ` with LAPACK `dpptrf` (packed storage), linked from OpenBLAS.
3. **Scaling** `H_ij = L_ij · sqrt(sqrt(|S_i·S_j|))` (i > j), `H_jj = L_jj · sqrt(|S_j|)`, with the one-point PSD `S`.
4. **H2Coeffs**: `V_i = Σ_j H_ij · (cos φ_j, sin φ_j)` with the random phases `φ`.

Afterwards a per-point inverse FFT produces the time series, which is scaled, rotated, quantised to int16 and written
to the `.bts` file. For a 35×124 box, steps 1–4 cost ~12 000 s in the stock binary; everything else costs seconds.

The bit-identity target is the **data block of the `.bts` file** (the header contains a timestamp), and, stricter, the
double-precision array `V` before the inverse FFT (`TS_DUMP_V`). Both are checked.

## 2. Reproducing the reference exactly

### 2.1 Coherence: K classes and the libmvec trap

The `exp` argument depends on a frequency and on the *bit pattern* of the distance `r_ij`. A box has only K distinct
patterns (35×35: 1 240; 35×124: 13 967), so `exp` is evaluated K times per frequency instead of `N(N+1)/2` times and
gathered into the matrix.

The subtle part is *which* `exp`: gfortran vectorises the reference loop with libmvec `_ZGVbN2v_exp` (SSE variant,
no FMA) for pairs of elements, while the last element of every odd-length column goes through scalar glibc `exp`
(`__exp_fma`). The two disagree by up to 3 ulp on 43 % of the arguments. `turbsim/kern/ts_kern.c` reproduces that
pattern exactly (pair or tail, vector-path boundary `|x| ≥ 708.39599609375`), calling the real glibc/libmvec
functions. The build script rejects objects that contain FMA instructions or unexpected libmvec calls.

### 2.2 Cholesky: dpptrf is a fixed FMA chain

OpenBLAS `dpptrf` computes, for every element `(i, c)` of `L`, a chain that does not depend on blocking or threads:

```
s = A_ic
for k = 1 .. c-1 (ascending):
    if L_ck != 0:  s = fma(-L_ck, L_ik, s)
L_cc = sqrt(s);  r = 1.0 / L_cc;  L_ic = s * r
```

This was verified against the real library with ctypes: 0 deviations in 3.16 million entries (n up to 1 225,
1–24 threads, including zero signs). Any schedule that preserves this per-element chain is bit-identical: left- or
right-looking, tiled, batched, on any number of CPU threads or GPU waves. Note: the skip of `L_ck == 0` is part of the
chain; dropping it only changes zero signs. A reference built against netlib BLAS (separate multiply and add) would
round differently; the target is OpenBLAS.

### 2.3 Certified no-op elimination (the main speed-up)

An FMA `s' = fma(-a, b, s)` whose rounded result is bitwise `s` can be skipped; by induction over `k` the chain stays
identical. Because the coherence decays exponentially with distance and frequency, most products are far below the
last bit of the accumulator. TurbSimGPU skips an FMA only if an integer rule proves it is a no-op
(`turbsim/kern/ts_skip.h`, one source for CPU and GPU):

```
skip  ⇔  e(a) + e(b) ≤ E(s) − 56
```

* `e(x)`: exponent field − 1023 of `x` (bit extraction only); exact zero → −2048; subnormal → −1023; Inf/NaN → +4096
  (never skip). Then `|x| < 2^(e(x)+1)` for every finite `x ≠ 0`.
* `E(s)`: `max(field(s), 1) − 1023`; Inf/NaN → −8192.
* Proof sketch: for normal `s` with `|s| ∈ [2^E, 2^(E+1))`, `|ab| < 2^(E−54) = ulp(s)/4`. That is less than half the gap
  to either neighbour, also when `s` is a power of two (gap below = `ulp(s)/2`), so round-to-nearest returns `s`. For
  subnormal or zero `s`, `|ab| < 2^−1076`, below half the subnormal spacing; a zero may change sign, which cannot reach
  `V` (`V` starts at +0 and is only added to).
* Counterexamples show that the obvious shortcuts are wrong: threshold 55
  (`a = 0x1.fffffffffffffp-500, b = 0x1.fffffffffffffp-555, s = 0x1p-1000`) and the real-valued test
  `fl(|a|·|b|)·2^55 ≤ |s|` (`a = 2^-500, b = 2^-575, s = 0x1.0000000000001p-1022`, the product underflows).

Checks: 0 violations in ~10¹⁰ certificates on real boxes, 7.8 million targeted boundary triples (binade edges,
subnormals, ±0, ±Inf) against glibc `fma`, and L, INFO and V bit-identical for 72 exactly computed chains.

The rule is applied to whole chunks of 8 `k`-columns using exponent **sketches**: `Sk[i][K] = max e(L_ik)` over the
chunk, written only after all columns of the chunk are final. If `Sk[i][K] + Sk[c][K] ≤ E(s) − 56`, the whole chunk is
a no-op. Sketches exist per tile (8×8), per 64 columns and per row, so that high frequencies, where almost everything
is skipped, are not limited by the scan. Executing is always exact; only skipping needs the certificate, so every
heuristic (when to check, how coarse) can affect speed but never bits.

Remaining FMA work, averaged over all frequencies (element-ideal): 35×35 box 4 %, 35×124 box 3 %, FAST.Farm high-res
box (f ≤ 1.19 Hz) 24 %.

## 3. GPU design (AMD RDNA4, HIP)

### 3.1 Frequency lanes

A wave32 computes the **same matrix element for 32 consecutive frequencies**; each lane is its own scalar chain, so
correctness is trivial. Masked lanes do not save FP64 cycles on RDNA4 (measured), so skipping must be wave-uniform: a
chunk is skipped only if the certificate holds for all 32 lanes. Neighbouring frequencies (Δf ≈ 0.007 Hz) have almost
identical skip patterns (union only 1–2 % above the per-lane ideal). Layout: `L[(group·N_packed + element)·32 + lane]`.

Kernels (`turbsim/gpu/tsg_chol.hip`): left-looking in panels of 8 columns; `tsg_k_bulk` (8×4 tile per wave, chunk
skipping with the three sketch levels), `tsg_k_diag` (8×8 diagonal block: sqrt, reciprocal, INFO), `tsg_k_panel`
(rows below the diagonal block, writes the sketches).

### 3.2 Only correctly rounded operations on the GPU

`v_fma_f64`, multiply, add, `sqrt` and `1.0/x` are correctly rounded on gfx1201 (0 deviations in 10⁸ random cases plus
2·10⁷ cases in the skip band); FP64 denormals are never flushed in HIP. Everything is compiled with
`-ffp-contract=off -fno-fast-math -fno-gpu-approx-transcendentals`.

**No transcendental function runs on the GPU.** GPU `exp/sin/cos` return different bits, so the CPU computes, per batch,
the exp tables (glibc/libmvec as in 2.1), the `sincos` of the phases (scalar glibc, as the reference) and the `S` values
per point class (`turbsim/kern/ts_gpu_prep.c`). The GPU then does the gather, the Cholesky, the scaling factors
`sqrt(sqrt(|S_a·S_b|))` per class pair, and H2Coeffs (one wave per row, `j` ascending, multiply and add separate,
starting at +0), and returns `V` slices that the CPU copies into place.

### 3.3 Batching and overlap

A batch holds `b ≤ B ≤ 8` groups of 32 frequencies (B from the VRAM budget: ≈ 256·N(N+1)/2 bytes per group,
2.6 GB for N = 4 341). The CPU prepares batch s+1 while the GPU computes batch s (double buffering). The batch size can
vary from batch to batch (`TS_GPU_STAPEL_S`): group boundaries stay at multiples of 32 frequencies from 0, and every
lane is independent, so batch size, pauses and parallel processes never change a bit.

### 3.4 Error path

If a matrix is not positive definite, the GPU reports the smallest failing frequency; TurbSim then recomputes exactly
that frequency on the CPU path, which aborts with the reference error text byte for byte
(`LAPACK_DPPTRF: Leading minor order … is not positive definite …`). A latent race in the parallel CPU loop (a thread
could stop smaller frequencies after a failure) is fixed by `patch_turbsim_fehlerpfad.py`, so the reported frequency
is always the smallest one, as in the serial reference.

## 4. CPU path (also bit-identical)

The same binary computes on the CPU when no GPU is available (`TS_GPU=0`, or a build without `GPU=ON`): OpenMP over
frequencies, the K-class coherence, the dpptrf chain with chunk skipping in scalar/AVX2/AVX-512 kernels
(`TS_CHOL_ISA=scalar|avx2|avx512`), H2Coeffs and the identity path (v/w components without coherence) in C, and a
parallel inverse FFT. The 35×124 box took ~980 s with 12 threads instead of ~12 000 s (measured before the CPU
skip kernels were added; with them the Cholesky is 2–3× faster, full boxes not re-measured).

## 5. The patch chain

The patches are idempotent Python scripts that edit `TSsubs.f90` and `TurbSim.f90` of a v3.5.3 tree; the order is
mandatory and enforced by the build script and the tests.

| Patch | File | Purpose |
|---|---|---|
| `instr` | TSsubs | timers for the frequency-loop blocks (`TS_ZEITEN`) |
| `omp353` | TSsubs | OpenMP over frequencies (thread-private work arrays) |
| `dump` | TurbSim | `TS_DUMP_V=<file>` writes V (double) before the IFFT |
| `zeiten` | TurbSim | wall clock per program phase |
| `plan` | TurbSim | `TS_PLAN=1`: print sizes after the grid is built and exit (used by the runner) |
| `gather` | TSsubs | K-class coherence with glibc/libmvec tables |
| `chol` | TSsubs | `LAPACK_pptrf` → bit-identical `ts_chol_spec` |
| `eye` | TSsubs | identity path (SCMod = NONE) in O(N) |
| `farm3` | TSsubs | one Cholesky for u, v, w when their IEC parameters are identical |
| `rest` | TSsubs | IFFT and mean-wind rotation in parallel per point |
| `h2c` | TSsubs | H2Coeffs in C, cache-friendly |
| `fehlerpfad` | TSsubs | deterministic error path (smallest failing frequency) |
| `gpu` | TSsubs | call `tsg_iec_lauf` (GPU) before the CPU loop; CPU fallback if the GPU declines |
| `gpunur` | TSsubs | `TS_GPU=nur`: refuse the CPU fallback (abort instead) |

Everything GPU-specific is inside `#ifdef TURBSIM_GPU`; without `GPU=ON` the source is the CPU build.

## 6. Thermal and memory safety (why the runner looks the way it does)

On the development machine, minutes of full GPU load at 255–300 W made the whole PC hang (two hangs at a junction
temperature of 93–94 °C). Measurements during endurance tests:

* The junction (hotspot) temperature follows the power within ~1 s and sits 24–31 °C above the edge temperature at
  210–255 W; in every pause it drops back within 1–2 s. A brake on the hotspot therefore oscillates between ~66 and
  ~96 °C and does not cap the peaks.
* The edge temperature is slow (+0.15 °C/s under load, −0.05 °C/s idle with a quiet fan) and is a good control
  variable. The runner's default profile pauses at 58 °C edge and resumes at 53 °C, which kept the hotspot ≤ 85 °C.
* `SIGSTOP` only stops the CPU side; work already queued on the GPU ran on for 4–6 s at full power. The emergency
  watchdog therefore kills the process (`SIGKILL`): the power dropped from 210 W to 19 W in under a second. The box
  is then retried once.

Memory: the RAM need is exact for the big arrays (`V`: NumSteps×N×3, `S` and phases: NumFreq×N×3, all doubles) plus
a margin for the rest; every box runs in a `systemd-run --user --scope` with `MemoryMax` and no swap, so a wrong
estimate kills one box, not the desktop.

#!/usr/bin/env bash
# Baut t1_chol (Pruefstand Stufe G1): C-Kerne wie build_turbsim_ref353.sh + HIP-Kern tsg_chol.hip.
set -euo pipefail
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERN="$HIER/../../kern"
B="${1:-$HIER/bau}"
mkdir -p "$B"
FL="-O2 -ffp-contract=off -fno-fast-math -fno-builtin-exp -fno-builtin-sqrt -fno-tree-vectorize -fPIE"
gcc $FL -c "$KERN/ts_kern.c" -o "$B/ts_kern.o"
gcc $FL -c "$KERN/ts_chol_scalar.c" -o "$B/ts_chol_scalar.o"
gcc $FL -mavx2 -mfma -c "$KERN/ts_chol_avx2.c" -o "$B/ts_chol_avx2.o"
gcc $FL -mavx512f -mavx512dq -mavx512vl -mfma -c "$KERN/ts_chol_avx512.c" -o "$B/ts_chol_avx512.o"
# Erst uebersetzen, dann linken: hipcc behandelt sonst die .o-Dateien als HIP-Quelltext (-x hip).
hipcc -O3 --offload-arch=gfx1201 -ffp-contract=off -fno-fast-math -fno-gpu-approx-transcendentals -std=c++17 \
      -c "$HIER/t1_chol.cpp" -o "$B/t1_chol.o"
hipcc --offload-arch=gfx1201 "$B/t1_chol.o" "$B/ts_kern.o" "$B/ts_chol_scalar.o" "$B/ts_chol_avx2.o" "$B/ts_chol_avx512.o" \
      -lmvec -lm -o "$HIER/t1_chol"
echo "gebaut: $HIER/t1_chol"

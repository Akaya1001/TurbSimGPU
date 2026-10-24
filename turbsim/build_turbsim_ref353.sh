#!/usr/bin/env bash
# Builds TurbSim from an OpenFAST v3.5.3 source tree with the bit-identical acceleration patches (CPU and, with GPU=ON,
# the AMD GPU path). See README.md.
#
# Usage:
#   ./turbsim/build_turbsim_ref353.sh <openfast-v3.5.3-worktree> [build-dir]
# Example:
#   git clone --branch v3.5.3 --depth 1 https://github.com/OpenFAST/openfast.git ~/src/openfast-v3.5.3
#   GPU=ON ./turbsim/build_turbsim_ref353.sh ~/src/openfast-v3.5.3 build/turbsim-gpu
#
# Environment: GPU (OFF | ON: whole IEC frequency loop on the GPU, requires ROCm/hipcc), GPU_ARCH (gfx1201),
#              GIT_DESCRIBE (v3.5.3: version string in the .bts/.sum header identical to stock v3.5.3), JOBS (2),
#              OPENMP (ON).
# The patches are applied IN PLACE to the worktree (idempotent; use a dedicated clone, not your development tree).
#
# Byte-identical runtime ERROR texts: OpenFAST embeds absolute source paths in some messages. If you compare failure
# output against a reference binary, build from a worktree at the same absolute path. Wind-field data are bit-identical
# regardless of the path.
set -euo pipefail

SRC="${1:?usage: build_turbsim_ref353.sh <openfast-v3.5.3-worktree> [build-dir]}"
BUILD="${2:-build/turbsim}"
HIER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_DESCRIBE="${GIT_DESCRIBE:-v3.5.3}"
JOBS="${JOBS:-2}"
OPENMP="${OPENMP:-ON}"
GPU="${GPU:-OFF}"

command -v cmake >/dev/null || { echo "ERROR: cmake not found." >&2; exit 1; }
[ -f "$SRC/modules/turbsim/src/TSsubs.f90" ] || { echo "ERROR: $SRC is not an OpenFAST source tree." >&2; exit 1; }
gfortran --version | head -1 | grep -q ' 13\.' || echo "WARNING: gfortran is not 13.x (reference: 13.3.0); bit-identity is only verified with 13.3."

# --- 1. Patches (alle idempotent; Reihenfolge ist zwingend) --------------
TS="$SRC/modules/turbsim/src/TSsubs.f90"
MAIN="$SRC/modules/turbsim/src/TurbSim.f90"
python3 "$HIER/patch_turbsim_instr.py"  "$TS"
python3 "$HIER/patch_turbsim_omp353.py" "$TS"
python3 "$HIER/patch_turbsim_dump.py"   "$MAIN"
python3 "$HIER/patch_turbsim_zeiten.py" "$MAIN"
python3 "$HIER/patch_turbsim_plan.py"   "$MAIN"
python3 "$HIER/patch_turbsim_gather.py" "$TS"
for extra in patch_turbsim_chol.py patch_turbsim_eye.py patch_turbsim_farm3.py patch_turbsim_rest.py patch_turbsim_h2c.py patch_turbsim_fehlerpfad.py patch_turbsim_gpu.py patch_turbsim_gpunur.py; do
   [ -f "$HIER/$extra" ] && python3 "$HIER/$extra" "$TS"
done

# --- 2. C-Kerne ------------------------------------------------------------
# Basis-x86-64 ohne FMA (wie das Fortran der Referenz); -ffp-contract=off
# verhindert Kontraktion der nicht fusionierten Operationen; -fno-builtin-*
# laesst exp/sqrt als echte glibc-Aufrufe stehen.
mkdir -p "$BUILD"
KERN_FL="-O2 -ffp-contract=off -fno-fast-math -fno-builtin-exp -fno-builtin-sqrt -fno-tree-vectorize -fPIE"
gcc $KERN_FL -c "$HIER/kern/ts_kern.c" -o "$BUILD/ts_kern.o"
gcc $KERN_FL -c "$HIER/kern/ts_chol_scalar.c" -o "$BUILD/ts_chol_scalar.o"
OBJS="$(cd "$BUILD" && pwd)/ts_kern.o $(cd "$BUILD" && pwd)/ts_chol_scalar.o"
# H2Coeffs: AVX2-Vektoren ohne FMA (kein -mfma!), damit mul und add getrennt runden wie die Referenz.
gcc -O3 -mavx2 -ffp-contract=off -fno-fast-math -fno-builtin-sincos -fno-builtin-sin -fno-builtin-cos -fPIE -c "$HIER/kern/ts_h2c.c" -o "$BUILD/ts_h2c.o"
OBJS="$OBJS $(cd "$BUILD" && pwd)/ts_h2c.o"
if [ -f "$HIER/kern/ts_chol_avx2.c" ]; then
   gcc $KERN_FL -mavx2 -mfma -c "$HIER/kern/ts_chol_avx2.c" -o "$BUILD/ts_chol_avx2.o"; OBJS="$OBJS $(cd "$BUILD" && pwd)/ts_chol_avx2.o"
fi
if [ -f "$HIER/kern/ts_chol_avx512.c" ]; then
   gcc $KERN_FL -mavx512f -mavx512dq -mavx512vl -mfma -c "$HIER/kern/ts_chol_avx512.c" -o "$BUILD/ts_chol_avx512.o"; OBJS="$OBJS $(cd "$BUILD" && pwd)/ts_chol_avx512.o"
fi
# GPU-Pfad (Stufe G3): CPU-Vorbereitung wie ts_h2c (glibc sincos skalar, keine Kontraktion), HIP-Motor mit den
# Bitgleichheits-Schaltern (T0.2/G0: sqrt, 1/x, mul, add, fma korrekt gerundet; keine Kontraktion).
FFLAGS_GPU=""
LIBS_GPU=""
if [ "$GPU" = ON ]; then
   command -v hipcc >/dev/null || { echo "ERROR: GPU=ON, but hipcc (ROCm) was not found." >&2; exit 1; }
   gcc -O2 -ffp-contract=off -fno-fast-math -fno-builtin-sincos -fno-builtin-sin -fno-builtin-cos -fno-tree-vectorize -fPIE \
       -c "$HIER/kern/ts_gpu_prep.c" -o "$BUILD/ts_gpu_prep.o"
   hipcc -O3 --offload-arch="${GPU_ARCH:-gfx1201}" -ffp-contract=off -fno-fast-math -fno-gpu-approx-transcendentals -fPIE -std=c++17 \
       -c "$HIER/gpu/tsg_host.hip" -o "$BUILD/tsg_host.o"
   OBJS="$OBJS $(cd "$BUILD" && pwd)/ts_gpu_prep.o $(cd "$BUILD" && pwd)/tsg_host.o"
   FFLAGS_GPU="-DTURBSIM_GPU"
   LIBS_GPU="-L$(hipconfig --path 2>/dev/null || echo /opt/rocm)/lib -lamdhip64 -lstdc++"
   if objdump -d "$BUILD/ts_gpu_prep.o" | grep -qE 'vfmadd|vfmsub|vfnmadd'; then echo "ERROR: FMA instructions in ts_gpu_prep.o (would break bit-identity)" >&2; exit 1; fi
   if objdump -dr "$BUILD/ts_gpu_prep.o" | grep -q '_ZGV'; then echo "ERROR: libmvec calls in ts_gpu_prep.o (would break bit-identity)" >&2; exit 1; fi
fi
# Nachzaehlen: ausserhalb der SIMD-Kerne darf kein FMA entstehen.
if objdump -d "$BUILD/ts_kern.o" "$BUILD/ts_chol_scalar.o" "$BUILD/ts_h2c.o" | grep -qE 'vfmadd|vfmsub|vfnmadd'; then echo "ERROR: FMA instructions in ts_kern.o/ts_chol_scalar.o/ts_h2c.o (would break bit-identity)" >&2; exit 1; fi

# --- 3. Bauen ------------------------------------------------------------------
# Objekte statt Bibliothek in CMAKE_EXE_LINKER_FLAGS (Linkreihenfolge, s. build_gpu.sh);
# -lmvec explizit, weil ts_kern.o _ZGVbN2v_exp referenziert.
cmake -S "$SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DDOUBLE_PRECISION=ON \
      -DOPENMP="$OPENMP" \
      -DBUILD_TESTING=OFF \
      -DGIT_DESCRIBE="$GIT_DESCRIBE" \
      -DCMAKE_Fortran_FLAGS="-ffp-contract=off $FFLAGS_GPU" \
      -DCMAKE_EXE_LINKER_FLAGS="$OBJS -lmvec -lm $LIBS_GPU" \
      >/dev/null
# CMake kennt die C-Objekte nicht als Abhaengigkeit: Neulink erzwingen, sonst bleibt
# bei geaendertem Kern das alte Binary stehen.
rm -f "$BUILD/modules/turbsim/turbsim"
cmake --build "$BUILD" --target turbsim -j "$JOBS"

BIN="$(cd "$BUILD" && pwd)/modules/turbsim/turbsim"
OBJ="$(find "$BUILD" -name 'TSsubs.f90.o' | head -1)"
echo "FMA instructions in TSsubs.f90.o: $(objdump -d "$OBJ" | grep -cE 'vfmadd|vfmsub|vfnmadd' || true) (must be 0)"
# libmvec-Pruefung: in den UMGEBAUTEN Routinen darf gfortran nichts vektorisieren (Referenz rechnet dort
# skalar sincos; die Kohaerenz-exp laeuft ueber ts_kern.c). GENERAL und API sind unveraendert und rufen
# libmvec wie die Referenz selbst.
LIBMVEC_NEU=$(objdump -dr "$OBJ" | awk '/^[0-9a-f]+ <.*>:$/{f=$2} /_ZGVbN2v_/{print f}' | grep -vcE 'calcfouriercoeffs_(api|general)' || true)
echo "libmvec calls in patched routines of TSsubs.f90.o: $LIBMVEC_NEU (must be 0)"
[ "$LIBMVEC_NEU" = 0 ] || { echo "ERROR: libmvec calls in patched routines - bit-identity at risk" >&2; exit 1; }
cat <<HINWEIS

Done. Binary:
  $BIN
Version in the .bts header: from OpenFAST-$GIT_DESCRIBE

Run:     OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 $BIN box.inp          (TS_GPU=nur: GPU only, no CPU fallback)
Plan:    python3 $HIER/gpu_windfelder.py <inputs...> --binary $BIN --ziel out --nur-planen
Verify:  python3 $HIER/bts_sha256.py box.bts      (compare with examples/REFERENCE_SHA256.txt)
         python3 $HIER/vergleich_bts.py ref.bts new.bts
HINWEIS

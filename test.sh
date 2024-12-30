#!/bin/bash

set -eox pipefail

export NINJA_SUMMARIZE_BUILD=1
export NINJA_STATUS="[%r jobs | %P %f/%t @ %o/s | %w | ETA %W ] "

# Replace `gfx1012` with your desired GPU target.
# WARNING: only fill in 1 GPU target, otherwise expect your build time to
# double/triple!
export AMDGPU_TARGETS="gfx1012"

cmake -B build -GNinja -S . -L \
	-DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc.pl \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGPU_TARGETS="$AMDGPU_TARGETS"

ninja -C build -j tests -v

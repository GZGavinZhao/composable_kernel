#!/bin/bash

set -eox pipefail

# Replace `gfx1012` with your desired GPU target.
# WARNING: only fill in 1 GPU target, otherwise expect your build time to
# double/triple!
export AMDGPU_TARGETS="gfx1032"

cmake -B build -S . -L \
	-DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGPU_TARGETS="$AMDGPU_TARGETS"

make -j tests

#!/bin/sh
# Build the pinned CAV23 stack without modifying upstream sources.
set -eu
root=/opt/saynt
jobs=${SAYNT_BUILD_JOBS:-1}

# Pinned source revisions.
cat > "$root/sources.txt" <<'SOURCES'
paynt https://github.com/randriu/paynt.git ba6a98c070f8934ecfede61437b5ecb5be4505b7
storm https://github.com/randriu/storm.git a3e7dffe3ddd401cd08c657addeba5e7faeb62e8
stormpy https://github.com/oiver12/stormpy.git cd1212bae68922aa4cce2b9af029fe9b7f1ce453
carl https://github.com/ths-rwth/carl.git b68f1b60f97c3fc629f61a24164ea73a6a1bedbd
pycarl https://github.com/moves-rwth/pycarl.git fcee9162a3428e8bc874e9cbff46f5ff32ccc886
SOURCES

# Download and verify the original sources.
mkdir -p "$root/source"
while read -r name repository revision; do
    git init "$root/source/$name"
    git -C "$root/source/$name" remote add origin "$repository"
    git -C "$root/source/$name" fetch --depth 1 origin "$revision"
    git -C "$root/source/$name" checkout --detach FETCH_HEAD
    test "$(git -C "$root/source/$name" rev-parse HEAD)" = "$revision"
done < "$root/sources.txt"
git -C "$root/source/storm" fetch --depth 1 origin refs/tags/1.6.4-cav23:refs/tags/1.6.4-cav23
test "$(git -C "$root/source/storm" rev-parse '1.6.4-cav23^{}')" = a3e7dffe3ddd401cd08c657addeba5e7faeb62e8

# Build carl and Storm. Keep Storm's upstream release LTO setting enabled.
cmake -S "$root/source/carl" -B "$root/carl-build" \
    -DCMAKE_BUILD_TYPE=Release -DTHREAD_SAFE=ON -DUSE_CLN_NUMBERS=OFF \
    -DUSE_GINAC=OFF -DEXCLUDE_TESTS_FROM_ALL=ON -DEXPORT_TO_CMAKE=OFF
cmake --build "$root/carl-build" --target lib_carl --parallel "$jobs"
# The legacy sources need cstdint pre-included with the container's GCC 12.
cmake -S "$root/source/storm" -B "$root/storm-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-include cstdint' \
    -DSTORM_CARL_DIR_HINT="$root/carl-build" -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF \
    -DSTORM_USE_CLN_RF=OFF -DSTORM_USE_CLN_EA=OFF -DSTORM_USE_LTO=ON \
    -DSTORM_USE_SPOT_SYSTEM=OFF -DSTORM_USE_SPOT_SHIPPED=OFF \
    -DSTORM_USE_GUROBI=OFF -DSTORM_EXCLUDE_TESTS_FROM_ALL=ON
cmake --build "$root/storm-build" \
    --target storm storm-main storm-synthesis storm-pomdp --parallel "$jobs"

# Build and install the matching Python bindings.
(
    cd "$root/source/pycarl"
    python setup.py build --build-base=build-cav23 build_ext \
        --carl-dir="$root/carl-build" --jobs="$jobs" bdist_wheel
    python -m pip install --no-deps dist/*.whl
)
(
    cd "$root/source/stormpy"
    python setup.py build_ext --storm-dir="$root/storm-build" \
        --disable-dft --disable-gspn --disable-pars --jobs="$jobs" bdist_wheel
    python -m pip install --no-deps dist/*.whl
)

# Check that SAYNT and its bindings load successfully.
python "$root/source/paynt/paynt.py" --help

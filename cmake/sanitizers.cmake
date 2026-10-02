# Convenience presets for command-line use:
#   cmake -B build -DCMAKE_BUILD_TYPE=Debug -C cmake/sanitizers.cmake
#
# This file is sourced only when passed with -C; the main CMakeLists.txt
# handles the flags via VECIDX_ENABLE_ASAN / VECIDX_ENABLE_TSAN options.

# Uncomment ONE of these:
# set(VECIDX_ENABLE_ASAN ON CACHE BOOL "")
# set(VECIDX_ENABLE_TSAN ON CACHE BOOL "")

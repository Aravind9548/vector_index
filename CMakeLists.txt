cmake_minimum_required(VERSION 3.20)
project(vecidx
  VERSION   0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(VECIDX_BUILD_TESTS      "Build unit tests (GoogleTest)"       ON)
option(VECIDX_BUILD_BENCHMARKS "Build micro-benchmarks (Google Benchmark)" ON)
option(VECIDX_ENABLE_ASAN      "Enable AddressSanitizer + UBSan"    OFF)
option(VECIDX_ENABLE_TSAN      "Enable ThreadSanitizer"             OFF)

# ---------------------------------------------------------------------------
# Compiler warnings — applied only to our own targets, not fetched deps
# ---------------------------------------------------------------------------
set(VECIDX_WARNING_FLAGS
  -Wall -Wextra -Wpedantic -Werror
  -Wconversion -Wsign-conversion -Wshadow
)

# ---------------------------------------------------------------------------
# Sanitizer presets (mutually exclusive — ASan and TSan cannot coexist)
# ---------------------------------------------------------------------------
if(VECIDX_ENABLE_ASAN AND VECIDX_ENABLE_TSAN)
  message(FATAL_ERROR "ASan and TSan cannot be enabled simultaneously.")
endif()

if(VECIDX_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(VECIDX_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

# ---------------------------------------------------------------------------
# Core library (header-only for now; will grow to a static lib)
# ---------------------------------------------------------------------------
add_library(vecidx_core INTERFACE)
target_include_directories(vecidx_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_compile_options(vecidx_core INTERFACE ${VECIDX_WARNING_FLAGS})

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

if(VECIDX_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # Windows CRT fix
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(tests)
endif()

if(VECIDX_BUILD_BENCHMARKS)
  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
  add_subdirectory(bench)
endif()

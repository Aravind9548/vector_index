# bench/CMakeLists.txt
add_executable(vecidx_bench
  flat_bench.cpp
)

target_link_libraries(vecidx_bench
  PRIVATE vecidx_core benchmark::benchmark_main
)

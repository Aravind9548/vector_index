# tests/CMakeLists.txt
add_executable(vecidx_tests
  smoke_test.cpp
  flat_index_test.cpp
)

target_link_libraries(vecidx_tests
  PRIVATE vecidx_core GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(vecidx_tests)

enable_testing()

add_executable(denox-common-test EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/all_minimum_cost_subgraph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/small_dynamic_bitset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/minimum_cost_subgraph.cpp
)

target_link_libraries(denox-common-test PRIVATE denox_common GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(denox-common-test)

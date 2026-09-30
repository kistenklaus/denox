
add_executable(denox-common-bench EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(denox-common-bench 
  PRIVATE denox_common benchmark::benchmark
)

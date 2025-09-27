include_guard(GLOBAL)  
include(cmake/colorful.cmake)

add_library(lewissbaker-generator INTERFACE)

log_success("✅ lewissbaker-generator available (local): ${CMAKE_SOURCE_DIR}/third-party/generator")
target_include_directories(lewissbaker-generator INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/generator/
)

add_library(denox::generator ALIAS lewissbaker-generator)

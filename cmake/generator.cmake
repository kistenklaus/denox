include_guard(GLOBAL)  

add_library(denox_generator INTERFACE)

target_include_directories(denox_generator INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/generator/
)

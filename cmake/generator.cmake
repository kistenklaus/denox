
add_library(lewissbaker_generator INTERFACE EXCLUDE_FROM_ALL)

target_include_directories(lewissbaker_generator INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/generator/
)

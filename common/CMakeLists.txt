include(${PROJECT_SOURCE_DIR}/cmake/fmt.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/spdlog.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/generator.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/warnings.cmake)

add_library(denox_common EXCLUDE_FROM_ALL
  src/denox/algorithm/binary_op_permutations.cpp
  src/denox/memory/dtype/dtype_reference.cpp
  src/denox/memory/dtype/f16.cpp
  src/denox/memory/tensor/ActivationTensor.cpp
  src/denox/memory/tensor/BiasTensor.cpp
  src/denox/memory/tensor/FilterTensor.cpp
  src/denox/io/read_file.cpp
  src/denox/io/fs/File.cpp
  src/denox/io/fs/Path.cpp
  src/denox/common/SHA256.cpp
)

target_include_directories(denox_common
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_common PUBLIC
    denox::generator
    denox::fmt
    denox::spdlog
)

target_compile_definitions(denox_common PUBLIC 
  DENOX_VERSION=0
)

target_compile_features(denox_common PUBLIC cxx_std_20)
set_target_properties(denox_common PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_common PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_common)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_common PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_common PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

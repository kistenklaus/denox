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
  src/denox/common/os.cpp
  src/denox/common/version.cpp
  src/denox/common/commit_hash.cpp

  src/denox/symbolic/SymGraph_add_entry.cpp
  src/denox/symbolic/SymGraph_affine.cpp
  src/denox/symbolic/SymGraph_affine_symbol_cache.cpp
  src/denox/symbolic/SymGraph_api.cpp
  src/denox/symbolic/SymGraph_compile.cpp
  src/denox/symbolic/SymGraph_debug.cpp
  src/denox/symbolic/SymGraph_div_entry.cpp
  src/denox/symbolic/SymGraph_eval.cpp
  src/denox/symbolic/SymGraph_helpers.cpp
  src/denox/symbolic/SymGraph_max_entry.cpp
  src/denox/symbolic/SymGraph_min_entry.cpp
  src/denox/symbolic/SymGraph_mod_entry.cpp
  src/denox/symbolic/SymGraph_modsolver.cpp
  src/denox/symbolic/SymGraph_modsolver_cache.cpp
  src/denox/symbolic/SymGraph_mul_entry.cpp
  src/denox/symbolic/SymGraph_nonaffine.cpp
  src/denox/symbolic/SymGraph_nonaffine_symbol_cache.cpp
  src/denox/symbolic/SymGraph_sub_entry.cpp
  src/denox/symbolic/SymGraphEval.cpp
  src/denox/symbolic/SymIR.cpp
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
  DENOX_VERSION_PATCH=${PROJECT_VERSION_PATCH}
  DENOX_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  DENOX_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
)

target_compile_features(denox_common PUBLIC cxx_std_20)
set_target_properties(denox_common PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_common PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_common PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (DENOX_UNITY_BUILD)
  set_target_properties(denox_common PROPERTIES UNITY_BUILD ON)
endif()


if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_common)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_common PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_common PRIVATE -fsanitize=undefined -fsanitize=address)
endif()


#  denox git commit hash 
find_package(Git QUIET)

set(DENOX_GIT_COMMIT_HASH "unknown")

if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE DENOX_GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/denox_commit_hash.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/denox_commit_hash.hpp
    @ONLY
)
target_include_directories(denox_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

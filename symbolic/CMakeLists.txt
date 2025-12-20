
add_library(denox_symbolic EXCLUDE_FROM_ALL
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
)

target_compile_features(denox_symbolic PUBLIC cxx_std_20)
set_target_properties(denox_symbolic PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_symbolic PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_symbolic PROPERTIES POSITION_INDEPENDENT_CODE ON)


target_include_directories(denox_symbolic
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_symbolic 
  PUBLIC 
    denox_common
)


if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_symbolic)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_symbolic PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_symbolic PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

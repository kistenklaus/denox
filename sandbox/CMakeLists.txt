
add_executable(sandbox EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_features(sandbox PUBLIC cxx_std_20)
set_target_properties(sandbox PROPERTIES CXX_STANDARD 20)
set_target_properties(sandbox PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(sandbox PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(sandbox PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sandbox 
  PUBLIC
    denox_compiler
    denox_common
    denox_spirv
    denox_database
    denox_runtime
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(sandbox)
endif()

if (DENOX_SAN) 
  target_compile_options(sandbox PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(sandbox PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

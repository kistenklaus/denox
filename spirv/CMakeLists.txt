
include(${PROJECT_SOURCE_DIR}/cmake/glslang.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/spirv-tools.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/warnings.cmake)

add_library(denox_spirv EXCLUDE_FROM_ALL
  src/denox/glsl/global_glslang_runtime.cpp
  src/denox/glsl/GlslPreprocessor.cpp
  src/denox/glsl/GlslCompiler.cpp
  src/denox/glsl/GlslCompilerInstance.cpp
)

target_compile_features(denox_spirv PUBLIC cxx_std_20)
set_target_properties(denox_spirv PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_spirv PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_spirv PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(denox_spirv
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_spirv PUBLIC
PRIVATE
  denox::glslang
PUBLIC
  denox_common
  denox_device_info
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_spirv)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_spirv PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_spirv PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

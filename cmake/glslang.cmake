include_guard(GLOBAL)
include(cmake/colorful.cmake)

# Helper: collect targets if they exist
function(_denox_collect_existing out_var)
  set(_candidates ${ARGN})
  set(_found "")
  foreach(t IN LISTS _candidates)
    if (TARGET ${t})
      list(APPEND _found ${t})
    endif()
  endforeach()
  set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

# 1) Prefer a proper CMake package (Vulkan SDK, distro, vcpkg, etc.)
find_package(glslang CONFIG QUIET)

if (glslang_FOUND)
  _denox_collect_existing(_DEN0X_GLSLANG_TGTS
    glslang::glslang
    glslang::SPIRV
    glslang::SPVRemapper
    glslang::OGLCompiler
    glslang::OSDependent
    glslang::HLSL
    glslang::glslang-default-resource-limits
    # legacy/un-namespaced fallbacks
    glslang SPIRV SPVRemapper OGLCompiler OSDependent HLSL glslang-default-resource-limits
  )

  if (_DEN0X_GLSLANG_TGTS)
    add_library(denox::glslang INTERFACE IMPORTED)
    target_link_libraries(denox::glslang INTERFACE ${_DEN0X_GLSLANG_TGTS})
    log_success("✅ glslang available (CMake config)")
    return()
  endif()
endif()

# 2) Fallback: fetch and build glslang from source (no binaries/tests)
include(FetchContent)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL ON CACHE BOOL "" FORCE)           # you likely want HLSL front-end available
set(ENABLE_OPT OFF CACHE BOOL "" FORCE)           # don't require SPIRV-Tools here
set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "" FORCE)
# Ensure PIC for static archives when linked into shared libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Pin to a tag you trust; adjust as you prefer.
FetchContent_Declare(denox_glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(denox_glslang)

# After add_subdirectory(), glslang exports proper targets:
# glslang::glslang, glslang::SPIRV, glslang::OGLCompiler, glslang::OSDependent, glslang::HLSL,
# glslang::SPVRemapper, glslang::glslang-default-resource-limits (names may vary slightly by version)
_denox_collect_existing(_DEN0X_GLSLANG_TGTS
  glslang::glslang
  glslang::SPIRV
  glslang::SPVRemapper
  glslang::OGLCompiler
  glslang::OSDependent
  glslang::HLSL
  glslang::glslang-default-resource-limits
)

if (NOT _DEN0X_GLSLANG_TGTS)
  message(FATAL_ERROR "glslang fetched but expected targets were not created")
endif()

add_library(denox::glslang INTERFACE IMPORTED)
target_link_libraries(denox::glslang INTERFACE ${_DEN0X_GLSLANG_TGTS})
log_success("✅ glslang available (fetched from source)")

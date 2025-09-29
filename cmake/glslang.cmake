# cmake/glslang.cmake
include_guard(GLOBAL)

# ---- Options --------------------------------------------------------------
# Prefer using a system-provided static lib first
option(DENOX_GLSLANG_FORCE_FETCH "Skip system lib; always fetch/build glslang static" OFF)
# Version to fetch (release tag). Change if you want a specific revision.
set(DENOX_GLSLANG_VERSION "15.4.0" CACHE STRING "glslang release tag to fetch when system lib not found")
# Optional integrity pin for the tarball (strongly recommended once known)
set(DENOX_GLSLANG_URL_HASH "" CACHE STRING "SHA256 of the glslang tarball (optional but recommended)")

# ---- Helper ---------------------------------------------------------------
function(_denox_collect_targets out_var)
  set(_found "")
  foreach(t IN LISTS ARGN)
    if (TARGET "${t}")
      list(APPEND _found "${t}")
    endif()
  endforeach()
  set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

# ---- 1) Try system static libglslang.a (+ headers) ------------------------
# if (NOT DENOX_GLSLANG_FORCE_FETCH)
#   # Prefer .a
#   set(_save_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
#   set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#   find_library(_GLSLANG_STATIC NAMES glslang)
#   set(CMAKE_FIND_LIBRARY_SUFFIXES "${_save_suffixes}")
#
#   find_path(_GLSLANG_INCLUDE_DIR
#     NAMES glslang/Public/ShaderLang.h
#     PATH_SUFFIXES include
#   )
#
#   if (_GLSLANG_STATIC AND _GLSLANG_INCLUDE_DIR)
#     add_library(denox_glslang STATIC IMPORTED GLOBAL)
#     set_target_properties(denox_glslang PROPERTIES
#       IMPORTED_LOCATION "${_GLSLANG_STATIC}"
#       INTERFACE_INCLUDE_DIRECTORIES "${_GLSLANG_INCLUDE_DIR}"
#     )
#
#     # Try to add companion libs if present (optional on modern glslang)
#     foreach(_n IN ITEMS SPIRV SPVRemapper OGLCompiler OSDependent HLSL glslang-default-resource-limits)
#       # prefer shared, then static
#       set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a")
#       find_library(_lib_${_n} NAMES ${_n})
#       if (_lib_${_n})
#         target_link_libraries(denox_glslang INTERFACE "${_lib_${_n}}")
#       endif()
#     endforeach()
#
#     add_library(denox::glslang ALIAS denox_glslang)
#     log_success("✅ glslang: using system STATIC core: ${_GLSLANG_STATIC} (headers: ${_GLSLANG_INCLUDE_DIR})")
#     return()
#   endif()
# endif()

# ---- 2) FetchContent from release tarball (static) ------------------------
include(FetchContent)

# Minimal & static build knobs
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING           OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL             OFF  CACHE BOOL "" FORCE)  # switch OFF if you don't need HLSL
set(ENABLE_OPT              OFF CACHE BOOL "" FORCE)  # keep OFF to avoid pulling SPIRV-Tools optimizer
set(SKIP_GLSLANG_INSTALL     ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_GLSLANG_URL "https://github.com/KhronosGroup/glslang/archive/refs/tags/${DENOX_GLSLANG_VERSION}.tar.gz")

# Use URL tarball to avoid git altogether (more robust in CI / behind proxies).
if (DENOX_GLSLANG_URL_HASH)
  FetchContent_Declare(denox_glslang_src
    URL       "${_GLSLANG_URL}"
    URL_HASH  "SHA256=${DENOX_GLSLANG_URL_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
else()
  # No hash pinned (less secure); still works. Pin once you've computed it.
  FetchContent_Declare(denox_glslang_src
    URL       "${_GLSLANG_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endif()

# Put sources in a stable path (avoids stale subbuilds)
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps")
FetchContent_MakeAvailable(denox_glslang_src)

# Upstream exports these targets; collect whatever exists on this tag.
_denox_collect_targets(_DEN0X_GLSLANG_TGTS
  glslang::glslang
  glslang::SPIRV
  glslang::SPVRemapper
  glslang::OGLCompiler
  glslang::OSDependent
  glslang::HLSL
  glslang::glslang-default-resource-limits
  # (Some versions also export legacy non-namespaced targets; add if needed.)
)

if (NOT _DEN0X_GLSLANG_TGTS)
  log_error("glslang (FetchContent): expected targets not created on version ${DENOX_GLSLANG_VERSION}")
endif()

add_library(denox::glslang INTERFACE IMPORTED)
target_link_libraries(denox::glslang INTERFACE ${_DEN0X_GLSLANG_TGTS})
log_success("✅ glslang: fetched ${DENOX_GLSLANG_VERSION} (static) via URL tarball")

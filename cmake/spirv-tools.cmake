include_guard(GLOBAL)
include(cmake/colorful.cmake)

option(DENOX_REQUIRE_SPIRV_TOOLS_OPT "Fail if SPIRV-Tools optimizer is not found" ON)

# Helper: collect existing targets
function(_collect out)
  set(acc "")
  foreach(t IN LISTS ARGN)
    if (TARGET "${t}")
      list(APPEND acc "${t}")
    endif()
  endforeach()
  set(${out} "${acc}" PARENT_SCOPE)
endfunction()

set(_core_libs "")
set(_opt_libs  "")

# 1) Prefer CMake config
find_package(SPIRV-Tools CONFIG QUIET)

if (SPIRV-Tools_FOUND)
  # Core targets (names vary by build)
  _collect(_core_targets
    SPIRV-Tools::SPIRV-Tools
    SPIRV-Tools                         # un-namespaced
    SPIRV-Tools-static                  # some builds
  )
  list(APPEND _core_libs ${_core_targets})

  # Optimizer targets (names vary)
  _collect(_opt_targets
    SPIRV-Tools-opt::SPIRV-Tools-opt
    SPIRV-Tools-opt
    SPIRV-Tools-opt-static
  )
  list(APPEND _opt_libs ${_opt_targets})
endif()

# 2) pkg-config fallback
if (_core_libs STREQUAL "" OR _opt_libs STREQUAL "")
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    if (_core_libs STREQUAL "")
      pkg_check_modules(SPVT IMPORTED_TARGET SPIRV-Tools QUIET)
      if (SPVT_FOUND)
        list(APPEND _core_libs PkgConfig::SPVT)
      endif()
    endif()
    if (_opt_libs STREQUAL "")
      pkg_check_modules(SPVT_OPT IMPORTED_TARGET SPIRV-Tools-opt QUIET)
      if (SPVT_OPT_FOUND)
        list(APPEND _opt_libs PkgConfig::SPVT_OPT)
      endif()
    endif()
  endif()
endif()

# 3) Direct library lookup (shared preferred, then static)
function(_find_lib out name)
  # Try shared first
  set(_save "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".dylib" ".dll.a" ".a")
  find_library(_loc NAMES "${name}")
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${_save}")
  set(${out} "${_loc}" PARENT_SCOPE)
endfunction()

if (_core_libs STREQUAL "")
  _find_lib(_loc_core "SPIRV-Tools")
  if (_loc_core)
    list(APPEND _core_libs "${_loc_core}")
  endif()
endif()

if (_opt_libs STREQUAL "")
  _find_lib(_loc_opt "SPIRV-Tools-opt")
  if (_loc_opt)
    list(APPEND _opt_libs "${_loc_opt}")
  endif()
endif()

# 4) Validate and create interface target
if (_core_libs STREQUAL "")
  log_error("❌ SPIRV-Tools core not found. Install 'spirv-tools' dev package or provide a CMake config/pkg-config.")
endif()

add_library(denox::spirv-tools INTERFACE IMPORTED)
target_link_libraries(denox::spirv-tools INTERFACE ${_core_libs})

if (NOT _opt_libs STREQUAL "")
  target_link_libraries(denox::spirv-tools INTERFACE ${_opt_libs})
  log_success("✅ SPIRV-Tools: core + optimizer")
else()
  if (DENOX_REQUIRE_SPIRV_TOOLS_OPT)
    log_error("❌ SPIRV-Tools optimizer (SPIRV-Tools-opt) not found but required.")
  else()
    log_warn("⚠️ SPIRV-Tools: core found, optimizer missing (SPIRV-Tools-opt).")
  endif()
endif()

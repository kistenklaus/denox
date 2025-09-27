include_guard(GLOBAL)
include(cmake/colorful.cmake)

# Try official CMake config first
find_package(SPIRV-Tools CONFIG QUIET)

set(_core_targets "")
set(_opt_targets  "")

if (SPIRV-Tools_FOUND)
  # Core (validator/as/dis)
  if (TARGET SPIRV-Tools::SPIRV-Tools)
    list(APPEND _core_targets SPIRV-Tools::SPIRV-Tools)
  elseif (TARGET SPIRV-Tools)
    list(APPEND _core_targets SPIRV-Tools)
  endif()

  # Optimizer (optional, if present)
  if (TARGET SPIRV-Tools-opt::SPIRV-Tools-opt)
    list(APPEND _opt_targets SPIRV-Tools-opt::SPIRV-Tools-opt)
  elseif (TARGET SPIRV-Tools-opt)
    list(APPEND _opt_targets SPIRV-Tools-opt)
  endif()
endif()

if (_core_targets STREQUAL "")
  find_package(PkgConfig QUIET)
  if (PkgConfig_FOUND)
    pkg_check_modules(SPVT     IMPORTED_TARGET SPIRV-Tools     QUIET)
    pkg_check_modules(SPVT_OPT IMPORTED_TARGET SPIRV-Tools-opt QUIET)

    if (SPVT_FOUND)
      list(APPEND _core_targets PkgConfig::SPVT)
    endif()
    if (SPVT_OPT_FOUND)
      list(APPEND _opt_targets PkgConfig::SPVT_OPT)
    endif()
  endif()
endif()

if (_core_targets STREQUAL "")
  log_error("❌ SPIRV-Tools not found. Install 'spirv-tools' (dev package) or provide a CMake config path.")
endif()

add_library(denox::spirv-tools INTERFACE IMPORTED)
target_link_libraries(denox::spirv-tools INTERFACE ${_core_targets})

if (NOT _opt_targets STREQUAL "")
  target_link_libraries(denox::spirv-tools INTERFACE ${_opt_targets})
  log_success("✅ SPIRV-Tools available (core + optimizer)")
else()
  log_warn("⚠️ SPIRV-Tools available (core only, optimizer missing)")
endif()

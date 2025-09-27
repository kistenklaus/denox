include_guard(GLOBAL)  

include(cmake/colorful.cmake)

find_package(Vulkan QUIET COMPONENTS shaderc_combined)

if (TARGET Vulkan::shaderc_combined)
  log_success("✅ shaderc available (system): ${Vulkan_shaderc_combined_LIBRARY}")
  add_library(denox::shaderc ALIAS Vulkan::shaderc_combined)
  return()
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(SHADERC IMPORTED_TARGET shaderc QUIET)
  if(SHADERC_FOUND)
    log_success("✅ shaderc (${SHADERC_VERSION}) available (pkg-config)")
    add_library(denox::shaderc ALIAS PkgConfig::SHADERC)
    return()
  endif()
endif()

log_error("❌ shaderc_combined not available!")

include_guard(GLOBAL)  
include(cmake/colorful.cmake)

# Options to control how we get fmt
option(DENOX_VENDOR_FMT "Download/build fmt if not found" ON)
option(DENOX_FMT_HEADER_ONLY "Use header-only fmt target" OFF)

set(DENOX_FMT_MIN_VERSION 9.1 CACHE STRING "Minimum fmt version")
set(DENOX_FMT_TAG "10.2.1" CACHE STRING "fmt tag to fetch if vendoring")

# Try system package first
find_package(fmt ${DENOX_FMT_MIN_VERSION} CONFIG QUIET)

if (fmt_FOUND) 
  if (DENOX_FMT_HEADER_ONLY)
    log_success("✅ fmt-header-only (${fmt_VERSION}) available (system): ${fmt_DIR}")
    add_library(denox::fmt ALIAS fmt::fmt-header-only)
  else()
    log_success("✅ fmt (${fmt_VERSION}) available (system) : ${fmt_DIR}")
    add_library(denox::fmt ALIAS fmt::fmt)
  endif()
  return()
endif()

if (NOT fmt_FOUND AND DENOX_VENDOR_FMT)
  include(FetchContent)
  # Avoid installing fmt along with us
  set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
  set(GITHUB_URL https://github.com/fmtlib/fmt/archive/refs/tags/${DENOX_FMT_TAG}.tar.gz)
  FetchContent_Declare(fmt
    URL ${GITHUB_URL}
    # or use GIT_REPOSITORY/GIT_TAG if you prefer
  )
  FetchContent_MakeAvailable(fmt)

  add_library(denox-fmt INTERFACE EXCLUDE_FROM_ALL)
  if (DENOX_FMT_HEADER_ONLY)
    log_success("✅ fmt-header-only available (system): ${GITHUB_URL}")
    target_link_libraries(denox-fmt INTERFACE fmt::fmt-header-only)
    return()
  else()
    log_success("✅ fmt available (FetchContent): ${GITHUB_URL}")
    target_link_libraries(denox-fmt INTERFACE fmt::fmt)
    # add_library(denox::fmt ALIAS fmt::fmt)
  endif()
  add_library(denox::fmt ALIAS denox-fmt)
  return()

endif()

log_error("❌ fmt not available!")

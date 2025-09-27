# cmake/deps/spdlog.cmake
include_guard(GLOBAL)
include(cmake/colorful.cmake)

option(DENOX_VENDOR_SPDLOG "Download/build spdlog if not found" ON)
# spdlog can be header-only or compiled; compiled speeds up large builds a bit
option(DENOX_SPDLOG_COMPILED "Prefer compiled spdlog target" OFF)

set(DENOX_SPDLOG_MIN_VERSION 1.10 CACHE STRING "Minimum spdlog version")
set(DENOX_SPDLOG_TAG "v1.14.1" CACHE STRING "spdlog tag to fetch if vendoring")

# 1) Try system/vcpkg/conan package
find_package(spdlog ${DENOX_SPDLOG_MIN_VERSION} CONFIG QUIET)

if (spdlog_FOUND)
  # Choose compiled if requested and available, else header-only if available, else compiled.
  if (DENOX_SPDLOG_COMPILED AND TARGET spdlog::spdlog)
    add_library(denox::spdlog ALIAS spdlog::spdlog)
    log_success("✅ spdlog-compiled (${spdlog_VERSION}) available (system) v${spdlog_DIR}")
  elseif (TARGET spdlog::spdlog_header_only)
    add_library(denox::spdlog ALIAS spdlog::spdlog_header_only)
    log_success("✅ spdlog (${spdlog_VERSION}) available (system) v${spdlog_DIR}")
  elseif (TARGET spdlog::spdlog)
    add_library(denox::spdlog ALIAS spdlog::spdlog)
    log_success("✅ spdlog-compiled (${spdlog_VERSION}) available (system) v${spdlog_DIR}")
  else()
    message(FATAL_ERROR "spdlog package found, but no known targets exported.")
  endif()
  return()
endif()

# 2) Fallback: FetchContent
if (DENOX_VENDOR_SPDLOG)
  include(FetchContent)

  # Make spdlog use external fmt (your denox::fmt should already exist)
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
  set(SPDLOG_COMPILED_LIB ${DENOX_SPDLOG_COMPILED} CACHE BOOL "" FORCE)
  set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/${DENOX_SPDLOG_TAG}.tar.gz
  )
  FetchContent_MakeAvailable(spdlog)

  add_library(denox-spdlog INTERFACE EXCLUDE_FROM_ALL)

  if (DENOX_SPDLOG_COMPILED AND TARGET spdlog::spdlog)
    target_link_libraries(denox-spdlog INTERFACE spdlog::spdlog)
    log_success("✅ spdlog-compiled available (FetchContent): ${DENOX_SPDLOG_TAG}")
  else()
    target_link_libraries(denox-spdlog INTERFACE spdlog::spdlog_header_only)
    log_success("✅ spdlog available (FetchContent): ${DENOX_SPDLOG_TAG}")
  endif()
  add_library(denox::spdlog ALIAS denox-spdlog)
  return()
endif()

message(FATAL_ERROR
  "spdlog >= ${DENOX_SPDLOG_MIN_VERSION} not found and vendoring is OFF")

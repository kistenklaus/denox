option(DENOX_VENDOR_SPDLOG "Download/build spdlog if not found" ON)
# spdlog is header-only by default; compiled lib can speed up your builds a bit.
option(DENOX_SPDLOG_COMPILED "Use compiled spdlog library target" OFF)
set(DENOX_SPDLOG_MIN_VERSION 1.10 CACHE STRING "Minimum spdlog version")
set(DENOX_SPDLOG_TAG "v1.14.1" CACHE STRING "spdlog tag to fetch if vendoring")

find_package(spdlog ${DENOX_SPDLOG_MIN_VERSION} CONFIG QUIET)
if (NOT spdlog_FOUND)
  if (DENOX_VENDOR_SPDLOG)
    include(FetchContent)
    # Make spdlog use our external fmt and (optionally) build compiled lib.
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
    set(SPDLOG_COMPILED_LIB ${DENOX_SPDLOG_COMPILED} CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(spdlog
      URL https://github.com/gabime/spdlog/archive/refs/tags/${DENOX_SPDLOG_TAG}.tar.gz
    )
    FetchContent_MakeAvailable(spdlog)
  else()
    message(FATAL_ERROR "spdlog >= ${DENOX_SPDLOG_MIN_VERSION} not found and vendoring is OFF")
  endif()
else()
  # If system spdlog was found, try to force external fmt (most distros already do).
  set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
endif()

# set(DENOX_SPDLOG_TARGET spdlog::spdlog_header_only) # header-only

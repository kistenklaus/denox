# Options to control how we get fmt
option(DENOX_VENDOR_FMT "Download/build fmt if not found" ON)
option(DENOX_FMT_HEADER_ONLY "Use header-only fmt target" OFF)
set(DENOX_FMT_MIN_VERSION 9.1 CACHE STRING "Minimum fmt version")
set(DENOX_FMT_TAG "10.2.1" CACHE STRING "fmt tag to fetch if vendoring")

# Try system package first
find_package(fmt ${DENOX_FMT_MIN_VERSION} CONFIG QUIET)

if (NOT fmt_FOUND)
  if (DENOX_VENDOR_FMT)
    include(FetchContent)
    # Avoid installing fmt along with us
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(fmt
      URL https://github.com/fmtlib/fmt/archive/refs/tags/${DENOX_FMT_TAG}.tar.gz
      # or use GIT_REPOSITORY/GIT_TAG if you prefer
    )
    FetchContent_MakeAvailable(fmt)
  else()
    message(FATAL_ERROR "fmt >= ${DENOX_FMT_MIN_VERSION} not found and vendoring is OFF")
  endif()
endif()

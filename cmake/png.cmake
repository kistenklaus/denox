# cmake/libpng.cmake
include_guard(GLOBAL)

# ---- Options --------------------------------------------------------------
option(DENOX_LIBPNG_FORCE_FETCH
  "Skip system libpng; always fetch/build libpng static" OFF)

set(DENOX_LIBPNG_VERSION "1.6.43"
  CACHE STRING "libpng release version to fetch when system lib not found")

set(DENOX_LIBPNG_URL_HASH ""
  CACHE STRING "SHA256 of libpng tarball (optional but recommended)")

# ---- 1) Try system libpng -------------------------------------------------
if (NOT DENOX_LIBPNG_FORCE_FETCH AND NOT DENOX_SAN)
  find_package(PNG QUIET)

  if (PNG_FOUND)
    if (TARGET PNG::PNG)
      add_library(denox::png ALIAS PNG::PNG)
    else()
      add_library(denox_png_system STATIC IMPORTED GLOBAL)
      set_target_properties(denox_png_system PROPERTIES
        IMPORTED_LOCATION "${PNG_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${PNG_INCLUDE_DIRS}"
      )
      add_library(denox::png ALIAS denox_png_system)
    endif()

    log_success("✅ libpng: using system package")
    return()
  endif()
endif()

# ---- 2) FetchContent fallback --------------------------------------------
include(FetchContent)

# libpng depends on zlib — prefer system zlib if available
find_package(ZLIB QUIET)

set(PNG_SHARED OFF CACHE BOOL "" FORCE)
set(PNG_STATIC ON  CACHE BOOL "" FORCE)
set(PNG_TESTS  OFF CACHE BOOL "" FORCE)
set(PNG_TOOLS  OFF CACHE BOOL "" FORCE)
set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(_LIBPNG_URL
  "https://download.sourceforge.net/libpng/libpng-${DENOX_LIBPNG_VERSION}.tar.gz")

if (DENOX_LIBPNG_URL_HASH)
  FetchContent_Declare(denox_libpng_src
    URL       "${_LIBPNG_URL}"
    URL_HASH  "SHA256=${DENOX_LIBPNG_URL_HASH}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
else()
  FetchContent_Declare(denox_libpng_src
    URL       "${_LIBPNG_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
endif()

FetchContent_MakeAvailable(denox_libpng_src)

# Upstream exports png_static or png
if (TARGET png_static)
  add_library(denox::png ALIAS png_static)
elseif (TARGET png)
  add_library(denox::png ALIAS png)
else()
  log_error("libpng (FetchContent): expected target not found")
endif()

log_success("✅ libpng: fetched ${DENOX_LIBPNG_VERSION} (static)")

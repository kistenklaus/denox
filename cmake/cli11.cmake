# cmake/denox_cli11.cmake
include_guard(GLOBAL)  # or DIRECTORY to scope to the including directory

include(${PROJECT_SOURCE_DIR}/cmake/colorful.cmake)
include(FetchContent)

# Allow pin override from the cache/CLI: -Ddenox_CLI11_TAG=v2.4.2 (or a commit)
set(denox_CLI11_TAG bfffd37e1f804ca4fae1caae106935791696b6a9 CACHE STRING "CLI11 version/commit to fetch")

# Progress + deterministic static builds for anything this dependency might build
# set(FETCHCONTENT_QUIET OFF CACHE BOOL "" FORCE)
# Requires CMake >= 3.27; harmless on older versions
# set(FETCHCONTENT_FULL_PROGRESS ON CACHE BOOL "" FORCE)
# set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build deps as static libs" FORCE)

# Keep CLI11 lean
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(CLI11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.1.tar.gz
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(CLI11)

if (TARGET CLI11::CLI11)
  log_success("✅ CLI11 available at: ${CLI11_SOURCE_DIR}")
  add_library(denox_cli11 INTERFACE)
  target_link_libraries(denox_cli11 INTERFACE CLI11::CLI11)
  add_library(denox::cli11 ALIAS denox_cli11)
else()
  log_error("❌ CLI11 target not available!")
endif()

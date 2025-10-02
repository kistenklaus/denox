include_guard(GLOBAL)
include(cmake/colorful.cmake)

include(FetchContent)
include(GoogleTest) # for gtest_discover_tests()

# Version pin (choose a concrete tag or commit)
set(DENOX_GTEST_TAG "v1.17.x" CACHE STRING "googletest tag to fetch")

# Windows: use same CRT as parent project
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Trim the build
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_build_tests OFF CACHE BOOL "" FORCE)
set(gtest_build_samples OFF CACHE BOOL "" FORCE)

set(DENOX_GTEST_REPO https://github.com/google/googletest.git)

FetchContent_Declare(
  googletest
  EXCLUDE_FROM_ALL
  GIT_REPOSITORY ${DENOX_GTEST_REPO}
  GIT_TAG        ${DENOX_GTEST_TAG}
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(googletest)

add_library(denox-gtest INTERFACE EXCLUDE_FROM_ALL)
add_library(denox-gtest-main INTERFACE EXCLUDE_FROM_ALL)

# Provide stable aliases for linking in your project
if (TARGET GTest::gtest)
  target_link_libraries(denox-gtest      INTERFACE GTest::gtest)
  target_link_libraries(denox-gtest-main INTERFACE GTest::gtest_main)
else()
  # Older layouts expose plain targets
  target_link_libraries(denox-gtest      INTERFACE gtest)
  target_link_libraries(denox-gtest-main INTERFACE gtest_main)
endif()

add_library(denox::gtest ALIAS denox-gtest)
add_library(denox::gtest-main ALIAS denox-gtest-main)

log_success("✅ googletest available (FetchContent) : ${DENOX_GTEST_REPO}:${DENOX_GTEST_TAG}")

enable_testing()

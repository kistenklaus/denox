include_guard(GLOBAL)

# always build flatbuffers runtime as a static runtime from source,
# but use local flatc if available.

# find_program(FLATC_EXECUTABLE flatc QUIET)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG 7e163021e59cca4f8e1e35a7c828b5c6b7915953 # v25.12.19
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(flatbuffers)

set_target_properties(flatc PROPERTIES UNITY_BUILD OFF)

set(FLATC_COMMAND $<TARGET_FILE:flatc>)


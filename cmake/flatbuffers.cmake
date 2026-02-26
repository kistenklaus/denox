include_guard(GLOBAL)

# always build flatbuffers runtime as a static runtime from source,
# but use local flatc if available.

find_program(FLATC_EXECUTABLE flatc QUIET)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)

if(FLATC_EXECUTABLE)
  execute_process(
    COMMAND ${FLATC_EXECUTABLE} --version
    OUTPUT_VARIABLE FLATC_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" FLATC_VERSION "${FLATC_VERSION_OUTPUT}")
  if(FLATC_VERSION)
    string(REPLACE "." ";" FLATC_VERSION_LIST ${FLATC_VERSION})
    list(GET FLATC_VERSION_LIST 0 FLATC_MAJOR)
    if(FLATC_MAJOR STREQUAL "25")
      set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "" FORCE)
    else()
      set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
    endif()
  else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
  endif()
else()
    set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
endif()

if (FLATBUFFERS_BUILD_FLATC)
  message(STATUS "Building flatc from source")
endif()

FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG 7e163021e59cca4f8e1e35a7c828b5c6b7915953 # v25.12.19
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(flatbuffers)

if (FLATBUFFERS_BUILD_FLATC)
  set_target_properties(flatc PROPERTIES UNITY_BUILD OFF)
endif()

if (FLATBUFFERS_BUILD_FLATC)
    # We built our own flatc
    set(FLATC_COMMAND $<TARGET_FILE:flatc>)
else()
    # Use system flatc
    set(FLATC_COMMAND ${FLATC_EXECUTABLE})
endif()


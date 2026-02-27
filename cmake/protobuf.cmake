include_guard(GLOBAL)

# Save current state (PAIN!!)
set(_old_unity ${CMAKE_UNITY_BUILD})
set(CMAKE_UNITY_BUILD OFF)
set(_old_c_flags   "${CMAKE_C_FLAGS}")
set(_old_cxx_flags "${CMAKE_CXX_FLAGS}")

if (MSVC)
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} /W0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
else()
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -w")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_CONFORMANCE OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # usually safer
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)         # enable if needed
set(protobuf_BUILD_LIBPROTOC OFF CACHE BOOL "" FORCE)
set(protobuf_FORCE_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(protobuf_USE_UNITY_BUILD ON CACHE BOOL "" FORCE) # best effort whatever that means?
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  protobuf
  GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
  GIT_TAG        b6f9284da830b69be787732ffdaa35049d20a088  # v33.5
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)


FetchContent_MakeAvailable(protobuf)

set(PROTOC_EXEC $<TARGET_FILE:protobuf::protoc>)

# Restore flags
set(CMAKE_UNITY_BUILD ${_old_unity})
set(CMAKE_C_FLAGS   "${_old_c_flags}")
set(CMAKE_CXX_FLAGS "${_old_cxx_flags}")

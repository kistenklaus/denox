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



set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)

set(SPIRV_TOOLS_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPIRV_TOOLS_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)

set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  spirv-headers
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
  GIT_TAG        04f10f650d514df88b76d25e83db360142c7b174 # from spirv-tools DEPS file of the used version
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)

FetchContent_Declare(
  spirv-tools
  GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git
  GIT_TAG        fbe4f3ad913c44fe8700545f8ffe35d1382b7093
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)

set(BUILD_EXTERNAL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
set(ENABLE_GLSLANG_JS OFF CACHE BOOL "" FORCE)
set(ENABLE_HLSL OFF CACHE BOOL "" FORCE)
# only possible because git hashes match exactly!
set(ALLOW_EXTERNAL_SPIRV_TOOLS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
  GIT_TAG        f0bd0257c308b9a26562c1a30c4748a0219cc951
  GIT_SHALLOW TRUE
  OVERRIDE_FIND_PACKAGE
  GIT_PROGRESS TRUE
  EXCLUDE_FROM_ALL
)

FetchContent_MakeAvailable(spirv-headers spirv-tools glslang)

add_library(denox_glslang INTERFACE)
target_link_libraries(denox_glslang 
  INTERFACE 
    glslang
    SPIRV-Tools
  )
# target_include_directories(denox_glslang
#     SYSTEM INTERFACE
#         $<TARGET_PROPERTY:glslang,INTERFACE_INCLUDE_DIRECTORIES>
# )

# Restore flags
set(CMAKE_UNITY_BUILD ${_old_unity})
set(CMAKE_C_FLAGS   "${_old_c_flags}")
set(CMAKE_CXX_FLAGS "${_old_cxx_flags}")

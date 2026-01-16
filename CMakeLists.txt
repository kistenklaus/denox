cmake_minimum_required(VERSION 3.24)

project(denox VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_PYTHON_BINDING "Build tests" OFF)
option(BUILD_TESTING "Build tests" ON)
option(DENOX_SAN "Enables sanitizers" OFF)
option(DENOX_DISABLE_WARNINGS "Enables sanitizers" OFF)
option(DENOX_UNITY_BUILD "Enable unity builds" OFF)
option(DENOX_USE_SYSTEM_FLATBUFFERS
  "Use system-installed FlatBuffers instead of FetchContent"
  ON)


include(cmake/fmt.cmake)
include(cmake/spdlog.cmake)
include(cmake/generator.cmake)
include(cmake/warnings.cmake)
include(cmake/vulkan.cmake)
include(cmake/glslang.cmake)
include(cmake/spirv-tools.cmake)
include(cmake/onnx.cmake)
include(cmake/dnx.cmake)
include(cmake/db.cmake)
include(cmake/vma.cmake)
include(cmake/png.cmake)

add_subdirectory(common)
add_subdirectory(device_info)
add_subdirectory(spirv)
add_subdirectory(database)
add_subdirectory(compiler)
add_subdirectory(runtime)
add_subdirectory(cli)
add_subdirectory(sandbox)

install(DIRECTORY
    ${CMAKE_BINARY_DIR}/share/denox
    DESTINATION share
)

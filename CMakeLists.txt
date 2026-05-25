cmake_minimum_required(VERSION 3.22)

project(denox VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TESTING "Build tests" ON)
option(DENOX_SAN "Enables sanitizers" OFF)

option(DENOX_ENABLE_NVML_CLOCKCTRL "Enables / Disables NVML clockctrl, this sometimes leads to ABI / compatibility issues on older GPUs" ON)

option(DENOX_STATIC_LIBSTDCPP "Enables statically linking libstdc++. Can improve ABI stability of produced binary" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fdiagnostics-color=always>
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
include(FetchContent)

# Dependencies build from source
include(cmake/fmt.cmake)
include(cmake/vma.cmake)
include(cmake/generator.cmake)
include(cmake/onnx.cmake)
include(cmake/flatbuffers.cmake)
include(cmake/dnx.cmake)
include(cmake/sqlite.cmake)
include(cmake/vulkan.cmake)
include(cmake/glslang.cmake)
include(cmake/protobuf.cmake)
include(cmake/png.cmake)
include(cmake/yaml-cpp.cmake)

# warnings helper
include(cmake/warnings.cmake)

add_subdirectory(common)
add_subdirectory(device_info)
add_subdirectory(spirv)
add_subdirectory(database)
add_subdirectory(runtime)
add_subdirectory(compiler)
add_subdirectory(cli)
add_subdirectory(sandbox)

install(DIRECTORY
    ${CMAKE_BINARY_DIR}/share/denox
    DESTINATION share
)

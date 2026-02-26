cmake_minimum_required(VERSION 4.2)

project(denox VERSION 0.1.0 LANGUAGES C CXX)

option(BUILD_TESTING "Build tests" ON)
option(DENOX_SAN "Enables sanitizers" OFF)
option(DENOX_DISABLE_WARNINGS "Disable pedantic warnings" OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-fdiagnostics-color=always>
    )
endif()

include(cmake/fmt.cmake)
include(cmake/generator.cmake)
include(cmake/warnings.cmake)
include(cmake/vulkan.cmake)
include(cmake/glslang.cmake)
include(cmake/spirv-tools.cmake)
include(cmake/onnx.cmake)
include(cmake/dnx.cmake)
include(cmake/vma.cmake)
include(cmake/png.cmake)
include(cmake/sqlite.cmake)

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

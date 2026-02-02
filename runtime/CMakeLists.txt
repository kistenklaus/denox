### Copy shaders into build directory
set(RUNTIME_SHADER_FILES 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/warmup.comp
)

add_custom_target(denox_runtime_shaders ALL)

foreach(shader IN LISTS RUNTIME_SHADER_FILES)
    file(RELATIVE_PATH rel
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${shader}
    )

    add_custom_command(
        TARGET denox_runtime_shaders
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${shader}
                ${CMAKE_BINARY_DIR}/share/denox/compiler/${rel}
        VERBATIM
    )
endforeach()

add_library(denox_runtime EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/vma_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/instance.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/db.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/clockctrl/clockctrl.cpp
)

target_compile_features(denox_runtime PUBLIC cxx_std_20)
set_target_properties(denox_runtime PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_runtime PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (DENOX_UNITY_BUILD)
  set_target_properties(denox_runtime PROPERTIES UNITY_BUILD ON)
endif()

target_include_directories(denox_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_runtime 
PUBLIC
  denox::vulkan
  denox::dnx
  denox::vma
  denox_device_info
  denox_spirv
  denox_common
  denox_database
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_runtime)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_runtime PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_runtime PRIVATE -fsanitize=undefined -fsanitize=address)
endif()


target_compile_definitions(denox_runtime PRIVATE GPU_CLOCKCTRL_HAS_CUDA)

find_path(NVML_INCLUDE_DIR
    NAMES nvml.h
    PATH_SUFFIXES
        nvidia-ml
        include
    PATHS
        /usr
        /opt/cuda/targets/x86_64-linux
        /opt/cuda
)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML
    REQUIRED_VARS NVML_INCLUDE_DIR
)

if (NVML_FOUND)
    add_library(NVML::Headers INTERFACE IMPORTED)
    set_target_properties(NVML::Headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NVML_INCLUDE_DIR}"
    )
    target_link_libraries(denox_runtime PRIVATE NVML::Headers nvidia-ml)
    target_compile_definitions(denox_runtime PRIVATE DENOX_HAS_NVML=1)
endif()

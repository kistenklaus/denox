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


)

target_compile_features(denox_runtime PUBLIC cxx_std_20)
set_target_properties(denox_runtime PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_runtime PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

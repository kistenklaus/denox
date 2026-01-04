
add_library(denox_runtime EXCLUDE_FROM_ALL
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/vma_impl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/dnx_parse_helpers.cpp

  ${CMAKE_CURRENT_SOURCE_DIR}/src/denox/runtime/model.cpp

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
)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_runtime)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_runtime PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_runtime PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

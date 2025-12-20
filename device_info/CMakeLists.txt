include(${PROJECT_SOURCE_DIR}/cmake/vulkan.cmake)

add_library(denox_device_info 
  EXCLUDE_FROM_ALL
    src/denox/device_info/query/create_query_instance.cpp
    src/denox/device_info/query/query_coopmat_properties.cpp
    src/denox/device_info/query/query_driver_device_info.cpp
    src/denox/device_info/query/query_layout_rules.cpp
    src/denox/device_info/query/query_memory_model_properties.cpp
    src/denox/device_info/query/query_resource_limits.cpp
    src/denox/device_info/query/query_subgroup_properties.cpp
    src/denox/device_info/query/select_physical_device.cpp
)

target_compile_features(denox_device_info PUBLIC cxx_std_20)
set_target_properties(denox_device_info PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_device_info PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_device_info PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(denox_device_info PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(denox_device_info 
  PRIVATE
    denox::vulkan
  PUBLIC
    denox_common
)

target_compile_features(denox_device_info PUBLIC cxx_std_20)
set_target_properties(denox_device_info PROPERTIES CXX_STANDARD 20)
set_target_properties(denox_device_info PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(denox_device_info PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT DENOX_DISABLE_WARNINGS)
  denox_enable_strict_warnings(denox_device_info)
endif()

if (DENOX_SAN) 
  target_compile_options(denox_device_info PRIVATE -fsanitize=undefined -fsanitize=address)
  target_link_options(denox_device_info PRIVATE -fsanitize=undefined -fsanitize=address)
endif()

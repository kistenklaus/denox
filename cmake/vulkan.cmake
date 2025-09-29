include_guard(GLOBAL)  # or DIRECTORY

include(cmake/colorful.cmake)

find_package(Vulkan REQUIRED)

if (Vulkan_FOUND)
  log_success("✅ Vulkan available: ${Vulkan_LIBRARY}")
else()
  log_error("❌ Vulkan not available!")
endif()

add_library(denox::vulkan ALIAS Vulkan::Vulkan)

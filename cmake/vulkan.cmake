include_guard(GLOBAL) 

find_package(Vulkan REQUIRED)

if(NOT TARGET Vulkan::Vulkan)
  message(FATAL_ERROR "Vulkan::Vulkan target not found")
endif()


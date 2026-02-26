include_guard(GLOBAL)

set(VMA_IMPL
    ${CMAKE_BINARY_DIR}/generated/vma.cpp
)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/vma.cpp.in
    ${VMA_IMPL}
    COPYONLY
)

add_library(denox_vma STATIC ${VMA_IMPL})

target_include_directories(denox_vma 
  SYSTEM  # <- avoids warnings
  PUBLIC 
  ${PROJECT_SOURCE_DIR}/third_party/vma
)

target_compile_definitions(denox_vma PRIVATE
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

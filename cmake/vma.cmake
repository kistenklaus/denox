include_guard(GLOBAL)
include(cmake/colorful.cmake)
include(cmake/vulkan.cmake)

# Allow overriding the fetched version/repo from the parent project.
set(DENOX_VMM_GIT_REPOSITORY
    "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git"
    CACHE STRING "VMA git repository")
set(DENOX_VMM_GIT_TAG
    "1d8f600fd424278486eade7ed3e877c99f0846b1"  # latest release as of 2025-05-12
    CACHE STRING "VMA git tag/commit to fetch")

set(_vma_target "")
set(_vma_found FALSE)

# 1) Prefer a preinstalled package
find_package(VulkanMemoryAllocator QUIET CONFIG)
if(TARGET GPUOpen::VulkanMemoryAllocator)
  set(_vma_target GPUOpen::VulkanMemoryAllocator)
  set(_vma_found TRUE)
elseif(TARGET VulkanMemoryAllocator) # un-namespaced, just in case
  set(_vma_target VulkanMemoryAllocator)
  set(_vma_found TRUE)
endif()

# 2) Fallback: fetch and add_subdirectory
if(NOT _vma_found)
  include(FetchContent)
  # Do not build its samples/tests when using as a dependency
  set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
  set(VMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(vma
    GIT_REPOSITORY "${DENOX_VMM_GIT_REPOSITORY}"
    GIT_TAG        "${DENOX_VMM_GIT_TAG}"
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(vma)

  if(TARGET GPUOpen::VulkanMemoryAllocator)
    set(_vma_target GPUOpen::VulkanMemoryAllocator)
    set(_vma_found TRUE)
  elseif(TARGET VulkanMemoryAllocator)
    set(_vma_target VulkanMemoryAllocator)
    set(_vma_found TRUE)
  else()
    log_error(
      "❌ VulkanMemoryAllocator target not found after fetch. "
      "Repo: ${DENOX_VMM_GIT_REPOSITORY} tag: ${DENOX_VMM_GIT_TAG}")
  endif()

  log_success("✅ denox::vmm: fetched ${_vma_target} (${DENOX_VMM_GIT_TAG})")
else()
  log_success("✅ denox::vmm: found ${_vma_target} via find_package()")
endif()


# Public alias used by your code
add_library(denox_vma INTERFACE)
target_link_libraries(denox_vma INTERFACE ${_vma_target} denox::vulkan)
add_library(denox::vma ALIAS denox_vma)

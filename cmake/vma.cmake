include_guard(GLOBAL)

include(cmake/colorful.cmake)
include(cmake/vulkan.cmake)

# Allow overriding the fetched version from the parent project.
# NOTE: Must be a *tag*, because tarballs only exist for tags.
set(DENOX_VMM_TAG
    "v3.2.1"   # latest release as of 2025-05-12
    CACHE STRING "VMA version/tag to fetch")

set(_vma_target "")
set(_vma_found FALSE)

# ---------------------------------------------------------------------------
# 1) Prefer a preinstalled package
# ---------------------------------------------------------------------------
find_package(VulkanMemoryAllocator QUIET CONFIG)
if(TARGET GPUOpen::VulkanMemoryAllocator)
    set(_vma_target GPUOpen::VulkanMemoryAllocator)
    set(_vma_found TRUE)
elseif(TARGET VulkanMemoryAllocator) # fallback name
    set(_vma_target VulkanMemoryAllocator)
    set(_vma_found TRUE)
endif()

# ---------------------------------------------------------------------------
# 2) Fallback: Fetch VMA tarball and add_subdirectory
# ---------------------------------------------------------------------------
if(NOT _vma_found)
    include(FetchContent)

    # Turn off components we do not want
    set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(VMA_BUILD_TESTS   OFF CACHE BOOL "" FORCE)

    # Use GitHub release tarballs to avoid git clone races
    FetchContent_Declare(vma
        URL         "https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/${DENOX_VMM_TAG}.tar.gz"
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(vma)

    # VMA creates one of these targets depending on version
    if(TARGET GPUOpen::VulkanMemoryAllocator)
        set(_vma_target GPUOpen::VulkanMemoryAllocator)
        set(_vma_found TRUE)
    elseif(TARGET VulkanMemoryAllocator)
        set(_vma_target VulkanMemoryAllocator)
        set(_vma_found TRUE)
    else()
        log_error("❌ VulkanMemoryAllocator target not found after fetching tarball '${DENOX_VMM_TAG}'")
    endif()

    log_success("✅ denox::vmm: fetched ${_vma_target} (tag: ${DENOX_VMM_TAG})")
else()
    log_success("✅ denox::vmm: found ${_vma_target} via find_package()")
endif()

# ---------------------------------------------------------------------------
# 3) Public alias used by your code
# ---------------------------------------------------------------------------
add_library(denox_vma INTERFACE)
target_link_libraries(denox_vma INTERFACE ${_vma_target} denox::vulkan)
add_library(denox::vma ALIAS denox_vma)

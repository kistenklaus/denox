#pragma once

#include <algorithm>
#include <cstddef>
namespace denox::memory {

class mallocator {
public:
  static constexpr bool is_always_equals = true;

  void *allocate(std::size_t size, std::size_t align) {
    align = std::max(align, alignof(std::max_align_t));
    size = (size + align - 1) & ~(align - 1);
#ifdef _MSC_VER
    return _aligned_malloc(size, align);
#else
    return std::aligned_alloc(align, size);
#endif
  }

  void deallocate(void *ptr, std::size_t, std::size_t) { deallocate(ptr); }

  void deallocate(void *ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
  }
};

} // namespace denox::memory

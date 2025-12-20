#pragma once
#include <cassert>
#include <cstddef>

namespace denox::memory {

template <typename Upstream> class allocator_ref {
public:
  allocator_ref(Upstream *upstream) : m_upstream(upstream) {}

  void *allocate(std::size_t size, std::size_t align) {
    assert(m_upstream != nullptr);
    return m_upstream->allocate(size, align);
  }

  void deallocate(void *ptr, std::size_t size, std::size_t align) {
    assert(m_upstream != nullptr);
    m_upstream->deallocate(ptr, size, align);
  }

private:
  Upstream *m_upstream;
};

} // namespace denox::memory

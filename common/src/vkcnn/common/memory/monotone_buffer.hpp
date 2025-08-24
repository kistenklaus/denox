#pragma once

#include "vkcnn/common/memory/align.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <utility>
namespace vkcnn::container {

class monotone_buffer {
public:
  monotone_buffer() : m_buffer(nullptr), m_capacity(0), m_size(0) {}

  std::size_t allocate(std::size_t size, std::size_t align) {
    std::size_t offset = m_size;
    offset = align_up(offset, align);
    std::size_t newSize = offset + size;
    if (newSize > m_capacity) {
      reserve(newSize * 2);
    }
    return offset;
  }

  void reserve(std::size_t capacity) {
    if (m_capacity >= capacity) {
      return;
    }
    void *newBuffer = malloc(capacity);
    std::memcpy(newBuffer, m_buffer, m_size);
    void *oldBuffer = std::exchange(m_buffer, newBuffer);
    free(oldBuffer);
  }

  void *get(std::size_t offset) {
    return static_cast<std::byte *>(m_buffer) + offset;
  }

  const void *get(std::size_t offset) const {
    return static_cast<const std::byte *>(m_buffer) + offset;
  }

  void *get() { return m_buffer; }
  const void *get() const { return m_buffer; }
  std::size_t size() const { return m_size; }

private:
  void *m_buffer;
  std::size_t m_capacity;
  std::size_t m_size;
};

} // namespace vkcnn::container

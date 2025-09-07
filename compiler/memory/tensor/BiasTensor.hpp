#pragma once

#include "memory/container/span.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/dtype/dtype_reference.hpp"
#include "memory/tensor/BiasDescriptor.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <fmt/base.h>
#include <functional>
#include <memory>

namespace denox::memory {

class BiasTensor;
class BiasTensorConstView;

class BiasTensorView {
public:
  friend class BiasTensorConstView;
  explicit BiasTensorView(BiasDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit BiasTensorView(BiasTensor *tensor);
  BiasTensorView(BiasTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  denox::memory::span<const std::byte> span() const {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }
  denox::memory::span<std::byte> span() {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int c) {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  dtype_const_reference at(unsigned int c) const {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  dtype_reference operator[](unsigned int c) {
    std::size_t offset = c * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return dtype_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }
  dtype_const_reference operator[](unsigned int c) const {
    std::size_t offset = c * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  void assignFrom(const BiasTensorConstView &view);

private:
  BiasDescriptor m_desc;
  std::byte *m_buffer;
};

class BiasTensorConstView {
public:
  explicit BiasTensorConstView(BiasDescriptor desc, const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}
  explicit BiasTensorConstView(const BiasTensor *tensor);
  BiasTensorConstView(const BiasTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  denox::memory::span<const std::byte> span() const {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_const_reference at(unsigned int c) const {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.byteSize();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

  dtype_const_reference operator[](unsigned int c) const {
    std::size_t offset = c * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return dtype_const_reference(reinterpret_cast<const void *>(ptr),
                                 m_desc.type);
  }

private:
  BiasDescriptor m_desc;
  const std::byte *m_buffer;
};

class BiasTensor {
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit BiasTensor(BiasDescriptor desc,
                      denox::memory::span<const std::byte> bias,
                      const Alloc &alloc = {})
      : m_desc(desc) {
    using allocator_traits = std::allocator_traits<Alloc>;
    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    assert(n == bias.size());
    std::memcpy(ptr, bias.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
    requires(!std::same_as<Alloc, denox::memory::span<std::byte>>)
  explicit BiasTensor(BiasDescriptor desc, const Alloc &alloc = {})
      : m_desc(desc) {
    using allocator_traits = std::allocator_traits<Alloc>;
    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memset(ptr, 0, n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
    requires(!std::same_as<Alloc, denox::memory::span<std::byte>>)
  explicit BiasTensor(BiasTensorConstView view, const Alloc &alloc = {})
      : m_desc(view.desc()) {
    using allocator_traits = std::allocator_traits<Alloc>;
    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    std::memcpy(ptr, view.data(), n);
    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  const std::byte *data() const { return m_storage.get(); }
  std::byte *data() { return m_storage.get(); }

  denox::memory::span<const std::byte> span() const {
    return denox::memory::span{m_storage.get(), m_desc.byteSize()};
  }

  denox::memory::span<std::byte> span() {
    return denox::memory::span{m_storage.get(), m_desc.byteSize()};
  }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int c) { return BiasTensorView{this}.at(c); }

  dtype_const_reference at(unsigned int c) const {
    return BiasTensorConstView{this}.at(c);
  }

  dtype_reference operator[](unsigned int c) { return BiasTensorView{this}[c]; }

  dtype_const_reference operator[](unsigned int c) const {
    return BiasTensorConstView{this}[c];
  }

private:
  BiasDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace denox::compiler

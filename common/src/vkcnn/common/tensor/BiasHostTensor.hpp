#pragma once

#include "vkcnn/common/tensor/BiasDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cassert>
#include <concepts>
#include <cstring>
#include <fmt/base.h>
#include <functional>
#include <memory>
namespace vkcnn {

class BiasHostTensor;
class BiasHostTensorConstView;

class BiasHostTensorView {
public:
  friend class BiasHostTensorConstView;
  explicit BiasHostTensorView(BiasDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit BiasHostTensorView(BiasHostTensor *tensor);
  BiasHostTensorView(BiasHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }
  std::span<std::byte> span() { return std::span{m_buffer, m_desc.byteSize()}; }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int c) {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }

  fxx_const_reference at(unsigned int c) const {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  fxx_reference operator[](unsigned int c) {
    std::size_t offset = c * m_desc.type.size();
    std::byte *ptr = m_buffer + offset;
    return fxx_reference(reinterpret_cast<void *>(ptr), m_desc.type);
  }
  fxx_const_reference operator[](unsigned int c) const {
    std::size_t offset = c * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  void assignFrom(const BiasHostTensorConstView &view);

private:
  BiasDescriptor m_desc;
  std::byte *m_buffer;
};

class BiasHostTensorConstView {
public:
  explicit BiasHostTensorConstView(BiasDescriptor desc, const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}
  explicit BiasHostTensorConstView(const BiasHostTensor *tensor);
  BiasHostTensorConstView(const BiasHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_const_reference at(unsigned int c) const {
    std::size_t linearIndex = static_cast<std::size_t>(c);
    std::size_t offset = linearIndex * m_desc.byteSize();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

  fxx_const_reference operator[](unsigned int c) const {
    std::size_t offset = c * m_desc.type.size();
    const std::byte *ptr = m_buffer + offset;
    return fxx_const_reference(reinterpret_cast<const void *>(ptr),
                               m_desc.type);
  }

private:
  BiasDescriptor m_desc;
  const std::byte *m_buffer;
};

class BiasHostTensor {
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit BiasHostTensor(BiasDescriptor desc, std::span<const std::byte> bias,
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
    requires(!std::same_as<Alloc, std::span<std::byte>>)
  explicit BiasHostTensor(BiasDescriptor desc, const Alloc &alloc = {})
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

  const std::byte *data() const { return m_storage.get(); }
  std::byte *data() { return m_storage.get(); }

  std::span<const std::byte> span() const {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }

  std::span<std::byte> span() {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }

  const BiasDescriptor &desc() const { return m_desc; }
  unsigned int shape() const { return m_desc.shape; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int c) { return BiasHostTensorView{this}.at(c); }

  fxx_const_reference at(unsigned int c) const {
    return BiasHostTensorConstView{this}.at(c);
  }

  fxx_reference operator[](unsigned int c) {
    return BiasHostTensorView{this}[c];
  }

  fxx_const_reference operator[](unsigned int c) const {
    return BiasHostTensorConstView{this}[c];
  }

private:
  BiasDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace vkcnn

#pragma once

#include "memory/container/span.hpp"
#include "memory/dtype/dtype.hpp"
#include "memory/dtype/dtype_reference.hpp"
#include "memory/tensor/ActivationDescriptor.hpp"
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>

namespace denox::memory {

// fwd declarations
class ActivationTensor;
class ActivationTensorConstView;

class ActivationTensorView {
public:
  friend class ActivationTensorConstView;

  explicit ActivationTensorView(ActivationDescriptor desc, std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit ActivationTensorView(ActivationTensor *tensor);
  ActivationTensorView(ActivationTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  denox::memory::span<const std::byte> span() const {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }
  denox::memory::span<std::byte> span() {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int w, unsigned int h, unsigned int c);

  dtype_const_reference at(unsigned int w, unsigned int h,
                           unsigned int c) const;

  dtype_reference operator[](unsigned int linearIndex);

  dtype_const_reference operator[](unsigned int linearIndex) const;

  void assignFrom(const ActivationTensorConstView &view);

private:
  ActivationDescriptor m_desc;
  std::byte *m_buffer;
};

class ActivationTensorConstView {
public:
  explicit ActivationTensorConstView(ActivationDescriptor desc,
                                     const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  ActivationTensorConstView(ActivationTensorView view)
      : m_desc(view.m_desc), m_buffer(view.m_buffer) {}

  explicit ActivationTensorConstView(const ActivationTensor *tensor);
  ActivationTensorConstView(const ActivationTensor &tensor);

  const std::byte *data() const { return m_buffer; }

  denox::memory::span<const std::byte> span() const {
    return denox::memory::span{m_buffer, m_desc.byteSize()};
  }

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_const_reference at(unsigned int w, unsigned int h,
                           unsigned int c) const;

  dtype_const_reference operator[](unsigned int linearIndex) const;

private:
  ActivationDescriptor m_desc;
  const std::byte *m_buffer;
};

class ActivationTensor {
private:
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationTensor(ActivationDescriptor descriptor,
                            denox::memory::span<const std::byte> weights,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    assert(n == weights.size());
    std::memcpy(ptr, weights.data(), n);

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationTensor(ActivationDescriptor descriptor,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
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
  explicit ActivationTensor(ActivationDescriptor descriptor,
                            ActivationTensorConstView view,
                            const Alloc &alloc = {})
      : m_desc(descriptor) {
    using allocator_traits = std::allocator_traits<Alloc>;

    Alloc allocator = alloc;
    std::size_t n = m_desc.byteSize();
    std::byte *ptr = allocator_traits::allocate(allocator, n);
    if (shape() != view.shape()) {
      throw std::runtime_error("Invalid tensor shape! Shapes do not match!");
    }
    if (layout() == view.layout() && type() == view.type()) {
      std::memcpy(ptr, view.data(), n);
    } else {
      ActivationTensorView{m_desc, ptr}.assignFrom(view);
    }

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationTensor(ActivationTensorConstView view,
                            const Alloc &alloc = {})
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

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  Dtype type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  dtype_reference at(unsigned int w, unsigned int h, unsigned int c) {
    return ActivationTensorView{this}.at(w, h, c);
  }

  dtype_const_reference at(unsigned int w, unsigned int h,
                           unsigned int c) const {
    return ActivationTensorConstView{this}.at(w, h, c);
  }

  dtype_reference operator[](unsigned int linearIndex) {
    return ActivationTensorView{this}[linearIndex];
  }

  dtype_const_reference operator[](unsigned int linearIndex) const {
    return ActivationTensorConstView{this}[linearIndex];
  }

private:
  ActivationDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace denox::compiler

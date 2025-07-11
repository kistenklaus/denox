#pragma once

#include "vkcnn/common/tensor/ActivationDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <span>

namespace vkcnn {

// fwd declarations
class ActivationHostTensor;
class ActivationHostTensorConstView;

class ActivationHostTensorView {
public:
  friend class ActivationHostTensorConstView;

  explicit ActivationHostTensorView(ActivationDescriptor desc,
                                    std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  explicit ActivationHostTensorView(ActivationHostTensor *tensor);
  ActivationHostTensorView(ActivationHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }
  std::byte *data() { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }
  std::span<std::byte> span() { return std::span{m_buffer, m_desc.byteSize()}; }

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int w, unsigned int h, unsigned int c);

  fxx_const_reference at(unsigned int w, unsigned int h, unsigned int c) const;

  fxx_reference operator[](unsigned int linearIndex);

  fxx_const_reference operator[](unsigned int linearIndex) const;

  void assignFrom(const ActivationHostTensorConstView &view);

private:
  ActivationDescriptor m_desc;
  std::byte *m_buffer;
};

class ActivationHostTensorConstView {
public:
  explicit ActivationHostTensorConstView(ActivationDescriptor desc,
                                         const std::byte *buffer)
      : m_desc(desc), m_buffer(buffer) {}

  ActivationHostTensorConstView(ActivationHostTensorView view)
      : m_desc(view.m_desc), m_buffer(view.m_buffer) {}

  explicit ActivationHostTensorConstView(const ActivationHostTensor *tensor);
  ActivationHostTensorConstView(const ActivationHostTensor &tensor);

  const std::byte *data() const { return m_buffer; }

  std::span<const std::byte> span() const {
    return std::span{m_buffer, m_desc.byteSize()};
  }

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_const_reference at(unsigned int w, unsigned int h, unsigned int c) const;

  fxx_const_reference operator[](unsigned int linearIndex) const;

private:
  ActivationDescriptor m_desc;
  const std::byte *m_buffer;
};

class ActivationHostTensor {
private:
public:
  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationHostTensor(ActivationDescriptor descriptor,
                                std::span<const std::byte> weights,
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
  explicit ActivationHostTensor(ActivationDescriptor descriptor,
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
  explicit ActivationHostTensor(ActivationDescriptor descriptor,
                                ActivationHostTensorConstView view,
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
      ActivationHostTensorView{m_desc, ptr}.assignFrom(view);
    }

    m_storage = std::unique_ptr<std::byte[], std::function<void(std::byte *)>>(
        ptr, [allocator, n](std::byte *p) mutable {
          allocator_traits::deallocate(allocator, p, n);
        });
  }

  template <typename Alloc = std::allocator<std::byte>>
  explicit ActivationHostTensor(ActivationHostTensorConstView view,
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

  std::span<const std::byte> span() const {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }
  std::span<std::byte> span() {
    return std::span{m_storage.get(), m_desc.byteSize()};
  }

  const ActivationDescriptor &desc() const { return m_desc; }
  const ActivationShape &shape() const { return m_desc.shape; }
  ActivationLayout layout() const { return m_desc.layout; }
  FloatType type() const { return m_desc.type; }
  std::size_t byteSize() const { return m_desc.byteSize(); }

  fxx_reference at(unsigned int w, unsigned int h, unsigned int c) {
    return ActivationHostTensorView{this}.at(w, h, c);
  }

  fxx_const_reference at(unsigned int w, unsigned int h, unsigned int c) const {
    return ActivationHostTensorConstView{this}.at(w, h, c);
  }

  fxx_reference operator[](unsigned int linearIndex) {
    return ActivationHostTensorView{this}[linearIndex];
  }

  fxx_const_reference operator[](unsigned int linearIndex) const {
    return ActivationHostTensorConstView{this}[linearIndex];
  }

private:
  ActivationDescriptor m_desc;
  std::unique_ptr<std::byte[], std::function<void(std::byte *)>> m_storage;
};

} // namespace vkcnn

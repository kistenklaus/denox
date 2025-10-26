#pragma once
#include "denox/common/types.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace pydenox {

struct Tensor {

  ~Tensor() { release(); }
  Tensor(const Tensor &o) {
    std::size_t size = o.byte_size();
    m_data = std::malloc(size);
    std::memcpy(m_data, o.m_data, size);
    m_batchSize = o.m_batchSize;
    m_height = o.m_height;
    m_width = o.m_width;
    m_channels = o.m_channels;
    m_dtype = o.m_dtype;
    m_layout = o.m_layout;
  }
  Tensor &operator=(const Tensor &o) {
    if (this == &o) {
      return *this;
    }
    std::size_t size = o.byte_size();
    m_data = std::malloc(size);
    std::memcpy(m_data, o.m_data, size);
    m_batchSize = o.m_batchSize;
    m_height = o.m_height;
    m_width = o.m_width;
    m_channels = o.m_channels;
    m_dtype = o.m_dtype;
    m_layout = o.m_layout;
    release();
    return *this;
  }
  Tensor(Tensor &&o)
      : m_data(std::exchange(o.m_data, nullptr)),
        m_batchSize(std::exchange(o.m_batchSize, 0)),
        m_height(std::exchange(o.m_height, 0)),
        m_width(std::exchange(o.m_width, 0)),
        m_channels(std::exchange(o.m_channels, 0)),
        m_dtype(std::exchange(o.m_dtype, denox::DataType::Auto)),
        m_layout(std::exchange(o.m_layout, denox::Layout::Undefined)) {}

  Tensor &operator=(Tensor &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    std::swap(m_data, o.m_data);
    std::swap(m_batchSize, o.m_batchSize);
    std::swap(m_height, o.m_height);
    std::swap(m_width, o.m_width);
    std::swap(m_channels, o.m_channels);
    std::swap(m_dtype, o.m_dtype);
    std::swap(m_layout, o.m_layout);
    return *this;
  }

  // If we fail to infer strides we just fail, no implicit assumptions.
  static Tensor from(pybind11::object object, denox::DataType dtype,
                     denox::Layout layout);

  // Always returns a HWC tensor.
  pybind11::object to() const;

  Tensor transform(denox::DataType new_dtype, denox::Layout new_layout) const;

  denox::DataType dtype() const { return m_dtype; }
  denox::Layout layout() const { return m_layout; }
  std::size_t batchSize() const { return m_batchSize; }
  std::size_t height() const { return m_height; }
  std::size_t width() const { return m_width; }
  std::size_t channels() const { return m_channels; }
  const void *data() const { return m_data; }
  std::size_t byte_size() const {
    return m_batchSize * m_height * m_width * m_channels * dtype_size(m_dtype);
  }

  void release() {
    if (m_data != nullptr) {
      std::free(m_data);
      m_data = nullptr;
      m_batchSize = 0;
      m_height = 0;
      m_width = 0;
      m_channels = 0;
      m_dtype = denox::DataType::Auto;
      m_layout = denox::Layout::Undefined;
    }
  }

private:
  static std::size_t dtype_size(denox::DataType dtype);

  Tensor(void *data, std::size_t N, std::size_t H, std::size_t W, std::size_t C,
         denox::DataType dtype, denox::Layout layout);

  void *m_data;
  std::size_t m_batchSize;
  std::size_t m_height;
  std::size_t m_width;
  std::size_t m_channels;
  denox::DataType m_dtype;
  denox::Layout m_layout;
};

} // namespace pydenox

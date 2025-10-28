#pragma once
#include "denox/common/types.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace pydenox {

struct Tensor {
  static Tensor from(pybind11::object object, denox::DataType dtype,
                     denox::Layout layout);

  static Tensor make(const void *data, std::size_t batchSize,
                     std::size_t height, std::size_t width,
                     std::size_t channels, denox::DataType dtype,
                     denox::Layout layout, int rank);

  static void define(pybind11::module_ &m);

  ~Tensor() { release(); }
  Tensor(const Tensor &o);
  Tensor &operator=(const Tensor &o);
  Tensor(Tensor &&o);
  Tensor &operator=(Tensor &&o);
  Tensor transform(denox::DataType new_dtype, denox::Layout new_layout) const;

  denox::DataType dtype() const { return m_dtype; }
  denox::Layout layout() const { return m_layout; }
  std::size_t batchSize() const { return m_batchSize; }
  std::size_t height() const { return m_height; }
  std::size_t width() const { return m_width; }
  std::size_t channels() const { return m_channels; }
  const void *data() const { return m_data; }
  std::size_t byte_size() const;
  int rank() const { return m_rank; }

  void release();

  std::size_t size(int dim) const;

  pybind11::capsule to_dlpack() const;

private:
  static std::size_t dtype_size(denox::DataType dtype);

  Tensor(void *data, std::size_t N, std::size_t H, std::size_t W, std::size_t C,
         denox::DataType dtype, denox::Layout layout, int rank);

  void *m_data;
  std::size_t m_batchSize;
  std::size_t m_height;
  std::size_t m_width;
  std::size_t m_channels;
  int m_rank;
  denox::DataType m_dtype;
  denox::Layout m_layout;
};

} // namespace pydenox

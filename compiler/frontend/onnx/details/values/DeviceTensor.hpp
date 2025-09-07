#pragma once

#include "frontend/onnx/details/values/Dtype.hpp"
#include "frontend/onnx/details/values/TensorShape.hpp"
#include "model/Tensor.hpp"
#include <cstddef>

namespace denox::onnx::details {

class DeviceTensor {
public:
  explicit DeviceTensor(std::size_t rank, compiler::Tensor handle);
  TensorShape shape() const;
  std::size_t rank() const { return m_rank; }
  const compiler::Tensor &handle() const { return m_handle; }
  compiler::Tensor &handle() { return m_handle; }

  const compiler::SymGraph *graph() const;
  bool sameHandleAs(const DeviceTensor &o) const;
  memory::optional<Dtype> type() const;

private:
  std::size_t m_rank; // i.e CHW or NCHW
  compiler::Tensor m_handle;
};

} // namespace denox::onnx::details

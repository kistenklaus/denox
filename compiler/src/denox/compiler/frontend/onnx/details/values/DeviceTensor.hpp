#pragma once

#include "denox/compiler/frontend/onnx/details/values/Dtype.hpp"
#include "denox/compiler/frontend/onnx/details/values/TensorShape.hpp"
#include "denox/compiler/frontend/model/Tensor.hpp"
#include <cstddef>

namespace denox::onnx::details {

class DeviceTensor {
public:
  explicit DeviceTensor(std::size_t rank, compiler::TensorHandle handle);
  TensorShape shape() const;
  std::size_t rank() const { return m_rank; }
  const compiler::TensorHandle &handle() const { return m_handle; }
  compiler::TensorHandle &handle() { return m_handle; }

  const compiler::SymGraph *graph() const;
  bool sameHandleAs(const DeviceTensor &o) const;
  memory::optional<Dtype> type() const;

private:
  std::size_t m_rank; // i.e CHW or NCHW
  compiler::TensorHandle m_handle;
};

} // namespace denox::onnx::details

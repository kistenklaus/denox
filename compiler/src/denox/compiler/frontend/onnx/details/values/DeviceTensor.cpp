#include "denox/compiler/frontend/onnx/details/values/DeviceTensor.hpp"
#include "denox/diag/unreachable.hpp"
#include "denox/memory/container/vector.hpp"

namespace denox::onnx::details {

DeviceTensor::DeviceTensor(std::size_t rank, compiler::Tensor handle)
    : m_rank(rank), m_handle(std::move(handle)) {
  assert(m_rank == 3 || m_rank == 4);
}

TensorShape DeviceTensor::shape() const {
  const auto &w = m_handle.width();
  const auto &h = m_handle.height();
  const auto &g = w.graph();
  assert(g);
  assert(g == h.graph());
  memory::vector<compiler::Symbolic> dims;
  dims.reserve(m_rank);
  if (m_rank == 4) {
    // NCHW
    dims.push_back(compiler::Symbolic{g, Sym::Const(1)});
  }
  dims.push_back(compiler::Symbolic{g, m_handle.channels()});
  dims.push_back(h);
  dims.push_back(w);
  return TensorShape{g, std::move(dims)};
}

const compiler::SymGraph *DeviceTensor::graph() const {
  assert(m_handle.width().graph());
  assert(m_handle.width().graph() == m_handle.height().graph());
  return m_handle.width().graph();
}

bool DeviceTensor::sameHandleAs(const DeviceTensor &o) const {
  return m_handle.id() == o.m_handle.id();
}

memory::optional<Dtype> DeviceTensor::type() const {
  TensorDataType dtype = m_handle.type();
  switch (dtype) {
  case TensorDataType::Auto:
    return std::nullopt;
  case TensorDataType::Float16:
    return Dtype::Float16;
  case TensorDataType::Float32:
    return Dtype::Float32;
    break;
  };
  diag::unreachable();
}

} // namespace denox::onnx::details

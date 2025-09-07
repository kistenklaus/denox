#pragma once

#include "frontend/onnx/details/values/TensorShape.hpp"
#include "memory/container/span.hpp"
#include "memory/container/vector.hpp"
#include "symbolic/Symbolic.hpp"

namespace denox::onnx::details {

class TensorViewDesc {
public:
  static TensorViewDesc Identity(const TensorShape &shape);
  memory::span<const compiler::Symbolic> strides() const;
  const compiler::Symbolic &offset() const;
  bool isConstant() const;
  std::size_t constIndexOf(memory::span<const std::uint64_t> idx) const;
  compiler::Symbolic
  indexOfSym(memory::span<const compiler::Symbolic> idx) const;

  compiler::Symbolic symIndexOf(memory::span<const std::uint64_t> idx) const;
  TensorViewDesc withOffset(const compiler::Symbolic &delta) const;
  TensorViewDesc permute(memory::span<const std::int64_t> perm) const;
  TensorViewDesc slice(std::size_t axis, const compiler::Symbolic &start,
                       const compiler::Symbolic &step) const;
  TensorViewDesc reverse(std::size_t axis,
                         const compiler::Symbolic &size) const;
  TensorViewDesc
  normalizeNegativeStrides(std::span<const compiler::Symbolic> shape) const;
  TensorViewDesc unsqueeze(size_t axis) const;
  TensorViewDesc squeeze(std::size_t axis) const;
  TensorViewDesc
  broadcastInDim(memory::span<const compiler::Symbolic> fromShape,
                 memory::span<const compiler::Symbolic> toShape,
                 memory::span<const std::int64_t> axesMap) const;
  bool isRowMajorContiguous(memory::span<const std::uint64_t> dims) const;
  bool hasNegativeStride() const;
  std::size_t rank() const { return m_strides.size(); }

private:
  static bool isPermutation(memory::span<const std::int64_t> p, std::size_t r);
  explicit TensorViewDesc(std::vector<compiler::Symbolic> strides,
                          compiler::Symbolic offset);

private:
  memory::vector<compiler::Symbolic> m_strides;
  compiler::Symbolic m_offset;
};

} // namespace denox::onnx::details

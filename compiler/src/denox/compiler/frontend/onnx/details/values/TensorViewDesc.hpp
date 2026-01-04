#pragma once

#include "denox/compiler/frontend/onnx/details/values/TensorShape.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/Symbolic.hpp"

namespace denox::onnx::details {

class TensorViewDesc {
public:
  static TensorViewDesc Identity(const TensorShape &shape);
  memory::span<const Symbolic> strides() const;
  const Symbolic &offset() const;
  bool isConstant() const;
  std::size_t constIndexOf(memory::span<const std::uint64_t> idx) const;
  Symbolic
  indexOfSym(memory::span<const Symbolic> idx) const;

  Symbolic symIndexOf(memory::span<const std::uint64_t> idx) const;
  TensorViewDesc withOffset(const Symbolic &delta) const;
  TensorViewDesc permute(memory::span<const std::int64_t> perm) const;
  TensorViewDesc slice(std::size_t axis, const Symbolic &start,
                       const Symbolic &step) const;
  TensorViewDesc reverse(std::size_t axis,
                         const Symbolic &size) const;
  TensorViewDesc
  normalizeNegativeStrides(std::span<const Symbolic> shape) const;
  TensorViewDesc unsqueeze(size_t axis) const;
  TensorViewDesc squeeze(std::size_t axis) const;
  TensorViewDesc
  broadcastInDim(memory::span<const Symbolic> fromShape,
                 memory::span<const Symbolic> toShape,
                 memory::span<const std::int64_t> axesMap) const;
  bool isRowMajorContiguous(memory::span<const std::uint64_t> dims) const;
  bool hasNegativeStride() const;
  std::size_t rank() const { return m_strides.size(); }

private:
  static bool isPermutation(memory::span<const std::int64_t> p, std::size_t r);
  explicit TensorViewDesc(std::vector<Symbolic> strides,
                          Symbolic offset);

private:
  memory::vector<Symbolic> m_strides;
  Symbolic m_offset;
};

} // namespace denox::onnx::details

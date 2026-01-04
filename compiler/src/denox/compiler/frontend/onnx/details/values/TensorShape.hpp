#pragma once

#include "denox/memory/container/span.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include "denox/symbolic/Symbolic.hpp"

namespace onnx {
class TensorShapeProto;
}

namespace denox::onnx::details {

class TensorShape {
public:
  TensorShape(SymGraph *g, memory::vector<Symbolic> dims);
  TensorShape(SymGraph *g, memory::span<const Sym> dims);
  TensorShape(SymGraph *g, memory::span<const std::uint64_t> dims);
  TensorShape(SymGraph *g, memory::span<const std::int64_t> dims);

  memory::span<const Symbolic> dims() const;
  std::size_t rank() const { return m_dims.size(); }
  bool empty() const { return m_dims.empty(); }
  bool isConstant() const;
  bool hasSymbolic() const { return !isConstant(); }
  bool hasZeroDim() const;
  Symbolic numel() const;
  memory::vector<std::uint64_t> toU64() const;
  TensorShape permute(memory::span<const std::int64_t> perm) const;
  TensorShape unsqueeze(std::size_t axis) const;
  TensorShape squeeze(std::size_t axis) const;
  static TensorShape broadcast(const TensorShape &a, const TensorShape &b);
  const Symbolic &operator[](std::size_t i) const;
  Symbolic &operator[](std::size_t i) { return m_dims[i]; }
  SymGraph *graph() const { return m_graph; }

  static TensorShape parse(const ::onnx::TensorShapeProto &shape,
                           SymGraph *symGraph,
                           std::string_view tensorName = "<unnamed>");

private:
  static bool isPermutation(memory::span<const std::int64_t> p, std::size_t r);

private:
  SymGraph *m_graph;
  // NOTE: non negative!!!
  memory::vector<Symbolic> m_dims;
};

} // namespace denox::onnx::details

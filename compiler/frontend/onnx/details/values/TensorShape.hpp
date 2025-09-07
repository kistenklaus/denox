#pragma once

#include "memory/container/span.hpp"
#include "symbolic/SymGraph.hpp"
#include "symbolic/Symbolic.hpp"

namespace onnx {
class TensorShapeProto;
};

namespace denox::onnx::details {

class TensorShape {
public:
  TensorShape(compiler::SymGraph *g, memory::vector<compiler::Symbolic> dims);
  TensorShape(compiler::SymGraph *g, memory::span<const compiler::Sym> dims);
  TensorShape(compiler::SymGraph *g, memory::span<const std::uint64_t> dims);
  TensorShape(compiler::SymGraph *g, memory::span<const std::int64_t> dims);

  memory::span<const compiler::Symbolic> dims() const;
  std::size_t rank() const { return m_dims.size(); }
  bool empty() const { return m_dims.empty(); }
  bool isConstant() const;
  bool hasSymbolic() const { return !isConstant(); }
  bool hasZeroDim() const;
  compiler::Symbolic numel() const;
  memory::vector<std::uint64_t> toU64() const;
  TensorShape permute(memory::span<const std::int64_t> perm) const;
  TensorShape unsqueeze(std::size_t axis) const;
  TensorShape squeeze(std::size_t axis) const;
  static TensorShape broadcast(const TensorShape &a, const TensorShape &b);
  const compiler::Symbolic &operator[](std::size_t i) const;
  compiler::Symbolic &operator[](std::size_t i) { return m_dims[i]; }
  compiler::SymGraph *graph() const { return m_graph; }

  static TensorShape parse(const ::onnx::TensorShapeProto &shape,
                           compiler::SymGraph *symGraph,
                           std::string_view tensorName = "<unnamed>");

private:
  static bool isPermutation(memory::span<const std::int64_t> p, std::size_t r);

private:
  compiler::SymGraph *m_graph;
  // NOTE: non negative!!!
  memory::vector<compiler::Symbolic> m_dims;
};

} // namespace denox::onnx::details

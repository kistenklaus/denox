#include "denox/compiler/frontend/onnx/details/values/TensorViewDesc.hpp"

namespace denox::onnx::details {

TensorViewDesc TensorViewDesc::Identity(const TensorShape &shape) {
  auto graph = shape.graph();
  std::size_t r = shape.rank();
  memory::vector<compiler::Symbolic> strides(r);
  compiler::Symbolic offset{graph, Sym::Const(0)};
  if (r != 0) {
    strides[r - 1] = compiler::Symbolic{graph, Sym::Const(1)};
    for (std::size_t d = r - 1; d-- > 0;) {
      strides[d] = strides[d + 1] * shape.dims()[d + 1];
    }
  }
  return TensorViewDesc{std::move(strides), std::move(offset)};
}
memory::span<const compiler::Symbolic> TensorViewDesc::strides() const {
  return m_strides;
}
const compiler::Symbolic &TensorViewDesc::offset() const { return m_offset; }
bool TensorViewDesc::isConstant() const {
  if (m_offset.isSymbolic()) {
    return false;
  }
  for (const auto &s : m_strides) {
    if (s.isSymbolic()) {
      return false;
    }
  }
  return true;
}

std::size_t
TensorViewDesc::constIndexOf(memory::span<const std::uint64_t> idx) const {
  assert(isConstant());
  assert(idx.size() == m_strides.size());

  if (m_strides.empty()) {
    const std::int64_t off = m_offset.constant();
    assert(off >= 0);
    return static_cast<std::size_t>(off);
  }
  std::int64_t acc = m_offset.constant();
  for (std::size_t i = 0; i < idx.size(); ++i) {
    const std::int64_t si = m_strides[i].constant();
    acc += si * static_cast<std::int64_t>(idx[i]);
  }
  assert(acc >= 0);
  return static_cast<std::size_t>(acc);
}


compiler::Symbolic
TensorViewDesc::indexOfSym(memory::span<const compiler::Symbolic> idx) const {
  assert(idx.size() == m_strides.size());
  compiler::Symbolic acc = m_offset;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    acc = acc + (m_strides[i] * idx[i]);
  }
  return acc;
}
compiler::Symbolic
TensorViewDesc::symIndexOf(memory::span<const std::uint64_t> idx) const {
  assert(idx.size() == m_strides.size());
  compiler::Symbolic acc = m_offset;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    acc = acc + (m_strides[i] * static_cast<std::int64_t>(idx[i]));
  }
  return acc;
}
TensorViewDesc
TensorViewDesc::withOffset(const compiler::Symbolic &delta) const {
  return TensorViewDesc{m_strides, m_offset + delta};
}
TensorViewDesc
TensorViewDesc::permute(memory::span<const std::int64_t> perm) const {
  assert(isPermutation(perm, m_strides.size()));
  memory::vector<compiler::Symbolic> s(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i) {
    const auto p = static_cast<std::size_t>(perm[i]);
    assert(p < m_strides.size());
    s[i] = m_strides[p];
  }
  return TensorViewDesc{std::move(s), m_offset};
}
TensorViewDesc TensorViewDesc::slice(std::size_t axis,
                                     const compiler::Symbolic &start,
                                     const compiler::Symbolic &step) const {
  assert(axis < m_strides.size());
  if (step.isConstant()) {
    assert(step.constant() != 0 && "step must be nonzero");
  }
  auto s = m_strides;
  compiler::Symbolic off = m_offset + s[axis] * start;
  s[axis] = s[axis] * step; // may be negative
  return TensorViewDesc{std::move(s), std::move(off)};
}
TensorViewDesc TensorViewDesc::reverse(std::size_t axis,
                                       const compiler::Symbolic &size) const {
  assert(axis < m_strides.size());
  auto s = m_strides;
  compiler::Symbolic off = m_offset;
  if (!(size.isConstant() && size.constant() == 0)) {
    off = off + s[axis] * (size - 1);
    s[axis] = s[axis] * static_cast<std::int64_t>(-1);
  }
  return TensorViewDesc{std::move(s), std::move(off)};
}
TensorViewDesc TensorViewDesc::normalizeNegativeStrides(
    std::span<const compiler::Symbolic> shape) const {
  assert(shape.size() == m_strides.size());
  auto s = m_strides;
  compiler::Symbolic off = m_offset;
  for (size_t i = 0; i < s.size(); ++i) {
    // if stride < 0, move offset to front and flip stride
    // NOTE: works when stride sign is known (constant). If symbolic sign,
    // skip.
    if (s[i].isConstant() && s[i].constant() < 0) {
      off = off + s[i] * (shape[i] - 1); // s[i] is negative here
      s[i] = s[i] * static_cast<int64_t>(-1);
    }
  }
  return TensorViewDesc{std::move(s), std::move(off)};
}
TensorViewDesc TensorViewDesc::unsqueeze(size_t axis) const {
  auto s = m_strides;
  const std::size_t r = s.size();
  assert(axis <= r);
  compiler::Symbolic one{m_offset.graph(), Sym::Const(1)};
  compiler::Symbolic stride_here = (r == 0) ? one : (axis == r ? one : s[axis]);
  s.insert(s.begin() +
               static_cast<memory::vector<compiler::Symbolic>::difference_type>(
                   axis),
           stride_here);
  return TensorViewDesc{std::move(s), m_offset};
}
TensorViewDesc TensorViewDesc::squeeze(std::size_t axis) const {
  auto s = m_strides;
  assert(axis < s.size());
  s.erase(
      s.begin() +
      static_cast<memory::vector<compiler::Symbolic>::difference_type>(axis));
  return TensorViewDesc{std::move(s), m_offset};
}

TensorViewDesc
TensorViewDesc::broadcastInDim(memory::span<const compiler::Symbolic> fromShape,
                               memory::span<const compiler::Symbolic> toShape,
                               memory::span<const std::int64_t> axesMap) const {
  assert(axesMap.size() == fromShape.size());
  memory::vector<compiler::Symbolic> s(
      toShape.size(),
      compiler::Symbolic{m_offset.graph(), Sym::Const(0)});
  for (std::size_t i = 0; i < axesMap.size(); ++i) {
    std::size_t ax = static_cast<std::size_t>(axesMap[i]);
    assert(ax < s.size());
    // If fromShape[i]==1 and toShape[ax]>1 → stride 0; else copy stride
    if (fromShape[i].isConstant() && toShape[ax].isConstant()) {
      if (fromShape[i].constant() == 1 && toShape[ax].constant() > 1) {
        s[ax] = compiler::Symbolic{m_offset.graph(), Sym::Const(0)};
      } else {
        s[ax] = m_strides[i];
      }
    } else {
      // symbolic sizes: conservatively carry the stride; caller should ensure
      // correctness
      s[ax] = m_strides[i];
    }
  }
  return TensorViewDesc{std::move(s), m_offset};
}

bool TensorViewDesc::isRowMajorContiguous(
    memory::span<const std::uint64_t> dims) const {
  if (dims.size() != m_strides.size())
    return false;
  if (dims.empty())
    return true;
  for (const auto &st : m_strides)
    if (st.isSymbolic())
      return false;
  std::int64_t expect = 1;
  for (std::size_t i = dims.size(); i-- > 0;) {
    const std::int64_t si = m_strides[i].constant();
    if (si != expect)
      return false;
    expect *= static_cast<std::int64_t>(dims[i]);
  }
  return true;
}
bool TensorViewDesc::hasNegativeStride() const {
  for (const auto &st : m_strides) {
    if (st.isConstant() && st.constant() < 0)
      return true;
  }
  return false;
}
bool TensorViewDesc::isPermutation(memory::span<const std::int64_t> p,
                                   std::size_t r) {
  memory::vector<uint8_t> seen(r, 0);
  for (auto v : p) {
    if (v < 0)
      return false;
    std::size_t u = static_cast<std::size_t>(v);
    if (u >= r)
      return false;
    if (seen[u]++)
      return false; // dup
  }
  return p.size() == r; // redundant with your outer assert, but safe
}
TensorViewDesc::TensorViewDesc(std::vector<compiler::Symbolic> strides,
                               compiler::Symbolic offset)
    : m_strides(std::move(strides)), m_offset(std::move(offset)) {}
} // namespace denox::onnx::details

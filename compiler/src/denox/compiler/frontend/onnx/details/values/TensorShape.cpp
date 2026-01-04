#include "denox/compiler/frontend/onnx/details/values/TensorShape.hpp"
#include <cassert>
#include <onnx.pb.h>

namespace denox::onnx::details {

TensorShape::TensorShape(SymGraph *g, memory::vector<Symbolic> dims)
    : m_graph(std::move(g)), m_dims(std::move(dims)) {
  for (const auto &d : m_dims) {
    if (d.isConstant()) {
      assert(d.constant() >= 0);
    } else {
      assert(g != nullptr);
    }
  }
}
TensorShape::TensorShape(SymGraph *g, memory::span<const Sym> dims)
    : m_graph(g) {
  m_dims.reserve(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    Symbolic s{m_graph, dims[i]};
    if (s.isConstant()) {
      assert(s.constant() >= 0);
    } else {
      assert(g != nullptr);
    }
    m_dims.emplace_back(std::move(s));
  }
}
TensorShape::TensorShape(SymGraph *g, memory::span<const std::uint64_t> dims)
    : m_graph(g) {
  m_dims.reserve(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    m_dims.emplace_back(m_graph,
                        Sym::Const(static_cast<std::int64_t>(dims[i])));
  }
}
TensorShape::TensorShape(SymGraph *g, memory::span<const std::int64_t> dims)
    : m_graph(g) {
  m_dims.reserve(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    assert(dims[i] >= 0);
    m_dims.emplace_back(m_graph, Sym::Const(dims[i]));
  }
}
memory::span<const Symbolic> TensorShape::dims() const { return m_dims; }

bool TensorShape::isConstant() const {
  for (const auto &d : m_dims)
    if (d.isSymbolic())
      return false;
  return true;
}
bool TensorShape::hasZeroDim() const {
  for (const auto &d : m_dims)
    if (d.isConstant() && d.constant() == 0)
      return true;
  return false;
}
Symbolic TensorShape::numel() const {
  Symbolic one{m_graph, Sym::Const(1)};
  Symbolic n = one;
  for (const auto &d : m_dims) {
    if (d.isConstant() && d.constant() == 0)
      return Symbolic{m_graph, Sym::Const(0)};
    n = n * d;
  }
  return n;
}
memory::vector<std::uint64_t> TensorShape::toU64() const {
  assert(isConstant());
  memory::vector<std::uint64_t> out;
  out.reserve(m_dims.size());
  for (const auto &d : m_dims) {
    assert(d.isConstant());
    out.push_back(static_cast<std::uint64_t>(d.constant()));
  }
  return out;
}
TensorShape TensorShape::permute(memory::span<const std::int64_t> perm) const {
  assert(perm.size() == m_dims.size());
  assert(isPermutation(perm, m_dims.size()));
  memory::vector<Symbolic> v(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    const size_t p = static_cast<size_t>(perm[i]);
    v[i] = m_dims[p];
  }
  return TensorShape{m_graph, std::move(v)};
}
TensorShape TensorShape::unsqueeze(std::size_t axis) const {
  assert(axis <= m_dims.size());
  memory::vector<Symbolic> v = m_dims;
  v.insert(v.begin() +
               static_cast<memory::vector<Symbolic>::difference_type>(axis),
           Symbolic{m_graph, Sym::Const(1)});
  return TensorShape{m_graph, std::move(v)};
}
TensorShape TensorShape::squeeze(std::size_t axis) const {
  assert(axis < m_dims.size());
  memory::vector<Symbolic> v = m_dims;
  v.erase(v.begin() +
          static_cast<memory::vector<Symbolic>::difference_type>(axis));
  return TensorShape{m_graph, std::move(v)};
}
TensorShape TensorShape::broadcast(const TensorShape &a, const TensorShape &b) {
  auto g = a.m_graph;
  if (g == nullptr) {
    g = b.m_graph;
  }

  const size_t ra = a.rank(), rb = b.rank();
  const size_t r = (ra > rb ? ra : rb);
  memory::vector<Symbolic> out(r, Symbolic{g, Sym::Const(1)});

  for (size_t i = 0; i < r; ++i) {
    const bool haveA = (i < ra);
    const bool haveB = (i < rb);
    const Symbolic da =
        haveA ? a.m_dims[ra - 1 - i] : Symbolic{g, Sym::Const(1)};
    const Symbolic db =
        haveB ? b.m_dims[rb - 1 - i] : Symbolic{g, Sym::Const(1)};

    if (da.isConstant() && da.constant() == 1) {
      out[r - 1 - i] = db;
    } else if (db.isConstant() && db.constant() == 1) {
      out[r - 1 - i] = da;
    } else {
      if (da == db)
        out[r - 1 - i] = da;
      else
        throw std::runtime_error("broadcast: incompatible dimensions");
    }
  }
  return TensorShape{g, std::move(out)};
}
const Symbolic &TensorShape::operator[](std::size_t i) const {
  return m_dims[i];
}
bool TensorShape::isPermutation(memory::span<const std::int64_t> p,
                                std::size_t r) {
  if (p.size() != r)
    return false;
  memory::vector<std::uint8_t> seen(r, 0);
  for (auto v : p) {
    if (v < 0)
      return false;
    const std::size_t u = static_cast<std::size_t>(v);
    if (u >= r)
      return false;
    if (seen[u]++)
      return false;
  }
  return true;
}

TensorShape TensorShape::parse(const ::onnx::TensorShapeProto &shp,
                               SymGraph *symGraph,
                               std::string_view tensorName) {

  memory::vector<Symbolic> dims;
  dims.reserve(static_cast<std::size_t>(shp.dim_size()));
  auto g = symGraph;
  if (!g)
    throw std::runtime_error("vkcnn: symGraph is null");

  for (int i = 0; i < shp.dim_size(); ++i) {
    const auto &d = shp.dim(i);
    if (d.has_dim_value()) {
      const int64_t v = d.dim_value();
      if (v < 0) {
        throw std::runtime_error(fmt::format(
            "vkcnn: {} has negative dim at axis {}", tensorName, i));
      }
      dims.emplace_back(g, Sym::Const(v));
    } else if (d.has_dim_param()) {
      assert(symGraph != nullptr);
      const std::string &label = d.dim_param();
      if (label.empty()) {
        throw std::runtime_error(fmt::format(
            "vkcnn: {} has empty dim_param at axis {}", tensorName, i));
      }
      Sym s;
      if (shp.dim_size() == 4 && i == 0) {
        s = Sym::Const(1);
      } else {
        s = symGraph->var();
      }
      dims.emplace_back(g, s);
    } else {
      assert(symGraph != nullptr);
      Sym s = symGraph->var();
      dims.emplace_back(g, s);
    }
  }

  return TensorShape{g, std::move(dims)};
}
} // namespace denox::onnx::details

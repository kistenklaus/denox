#pragma once

#include "vkcnn/common/hypergraph/AdjGraph.hpp"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/symbolic/SymOp.hpp"
#include "vkcnn/common/symbolic/SymValue.hpp"
#include "vkcnn/common/symbolic/SymVar.hpp"
#include <glm/vec2.hpp>
#include <memory>
#include <unordered_map>

namespace vkcnn {

class SymAdjGraph {
public:
  SymAdjGraph() : m_store(std::make_shared<Storage>()) {}

  auto createParameter() { return createVar(); }

  auto ceilDiv(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::CeilDiv, lhs, rhs);
  }

  auto ceilDiv(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::CeilDiv, lhs, rhs);
  }

  auto ceilDiv(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::CeilDiv, lhs, rhs);
  }

  auto floorDiv(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::FloorDiv, lhs, rhs);
  }

  auto floorDiv(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::FloorDiv, lhs, rhs);
  }

  auto floorDiv(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::FloorDiv, lhs, rhs);
  }

  auto alignUp(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::AlignUp, lhs, rhs);
  }

  auto alignUp(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::AlignUp, lhs, rhs);
  }

  auto alignUp(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::AlignUp, lhs, rhs);
  }

  auto sub(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Sub, lhs, rhs);
  }

  auto sub(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::Sub, lhs, rhs);
  }

  auto sub(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Sub, lhs, rhs);
  }

  auto mul(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Mul, lhs, rhs);
  }

  auto mul(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::Mul, lhs, rhs);
  }

  auto mul(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Mul, lhs, rhs);
  }

  auto add(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Add, lhs, rhs);
  }

  auto add(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::Add, lhs, rhs);
  }

  auto add(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Add, lhs, rhs);
  }

  auto max(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Max, lhs, rhs);
  }

  auto max(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::Max, lhs, rhs);
  }

  auto max(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Max, lhs, rhs);
  }

  auto min(hypergraph::NodeId lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Min, lhs, rhs);
  }

  auto min(hypergraph::NodeId lhs, std::uint64_t rhs) {
    return binaryExpr(SymExpr::Min, lhs, rhs);
  }

  auto min(std::uint64_t lhs, hypergraph::NodeId rhs) {
    return binaryExpr(SymExpr::Min, lhs, rhs);
  }

  auto Conv2d(hypergraph::NodeId extent, unsigned int kernelSize,
              unsigned int padding, unsigned int stride,
              unsigned int dilation = 1) {

    const std::uint64_t ke_minus1 = dilation * (kernelSize - 1);
    if (stride == 1) {
      // NOTE: Easy case of "same" padding.
      if (2 * padding == ke_minus1) {
        return extent;
      }
      std::int64_t x0 = 2 * padding - ke_minus1;
      if (x0 < 0) {
        return sub(extent, static_cast<std::uint64_t>(-x0));
      } else {
        return add(extent, static_cast<std::uint64_t>(x0));
      }
    } else {
      // NOTE: Fallback to general formula.
      std::int64_t x0 = 2 * padding - ke_minus1 - 1;
      hypergraph::NodeId x1{0};
      if (x0 < 0) {
        x1 = sub(extent, static_cast<std::uint64_t>(-x0));
      } else if (x0 > 0) {
        x1 = add(extent, static_cast<std::uint64_t>(x0));
      }
      auto x2 = floorDiv(x1, static_cast<std::uint64_t>(stride));
      return add(x2, 1ull);
    }
  }

  auto Pool2d(hypergraph::NodeId extent, uint64_t kernel, uint64_t padding,
              uint64_t stride, uint64_t dilation = 1, bool ceilMode = false) {
    const uint64_t ke_minus1 = dilation * (kernel - 1);

    if (stride == 1) {
      // NOTE: Easy case of "same" padding.
      if (2ull * padding == ke_minus1)
        return extent;
      const std::int64_t c = 2 * padding - ke_minus1;
      if (c < 0) {
        return sub(extent, static_cast<std::uint64_t>(-c));
      } else {
        return add(extent, static_cast<std::uint64_t>(c));
      }
    }
    // NOTE: out = floor_or_ceil( (in + 2p - d*(k-1) - 1) / s ) + 1
    const int64_t c = 2 * padding - ke_minus1 - 1;
    hypergraph::NodeId t{0};
    if (c < 0) {
      t = sub(extent, static_cast<std::uint64_t>(-c));
    } else if (c > 0) {
      t = add(extent, static_cast<std::uint64_t>(c));
    } else {
      t = extent;
    }
    auto q = ceilMode ? ceilDiv(t, stride) : floorDiv(t, stride);
    return add(q, 1ull);
  }

  auto Upsample2d(hypergraph::NodeId extent, uint64_t scale) {
    if (scale == 1)
      return extent;
    return mul(extent, scale);
  }

  [[deprecated]]
  const hypergraph::AdjGraph<SymValue, SymExpr> &hypergraph() const {
    return m_store->hypergraph;
  }

  // NOTE: This does not invalidate SymVar
  hypergraph::ConstGraph<SymValue, SymExpr> freeze() const {
    return hypergraph::ConstGraph<SymValue, SymExpr>(m_store->hypergraph);
  }

private:
  hypergraph::NodeId createConstant(std::uint64_t c) {

    auto it = m_store->constantReuse.find(c);
    if (it == m_store->constantReuse.end()) {
      auto id = m_store->hypergraph.addNode(SymValue::Const(c));
      m_store->constantReuse.insert(it, {c, id});
      return id;
    } else {
      return it->second;
    }
  }

  inline hypergraph::NodeId createVar() {
    return m_store->hypergraph.addNode(SymValue::Var());
  }

  hypergraph::NodeId binaryExpr(SymExpr expr, hypergraph::NodeId lhs,
                                hypergraph::NodeId rhs) {
    auto var = createVar();
    m_store->hypergraph.addEdge(lhs, rhs, var, expr);
    return var;
  }

  hypergraph::NodeId binaryExpr(SymExpr expr, hypergraph::NodeId lhs,
                                std::uint64_t rhs) {
    // NOTE: Importantly c is created first, which ensures topological ordering.
    auto c = createConstant(rhs);
    auto var = createVar();
    m_store->hypergraph.addEdge(lhs, c, var, expr);
    return var;
  }

  hypergraph::NodeId binaryExpr(SymExpr expr, std::uint64_t lhs,
                                hypergraph::NodeId rhs) {
    // NOTE: Importantly c is created first, which ensures topological ordering.
    auto c = createConstant(lhs);
    auto var = createVar();
    m_store->hypergraph.addEdge(c, rhs, var, expr);
    return var;
  }

  struct Storage {
    hypergraph::AdjGraph<SymValue, SymExpr> hypergraph;
    std::unordered_map<std::uint64_t, hypergraph::NodeId> constantReuse;
  };
  std::shared_ptr<Storage> m_store;
};

} // namespace vkcnn

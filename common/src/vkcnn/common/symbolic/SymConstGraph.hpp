#pragma once

#include "ATen/core/interned_strings.h"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/hypergraph/topological_sort.hpp"
#include "vkcnn/common/memory/align.hpp"
#include "vkcnn/common/symbolic/SymAdjGraph.hpp"
#include "vkcnn/common/symbolic/SymEval.hpp"
#include "vkcnn/common/symbolic/SymIdMapping.hpp"
#include "vkcnn/common/symbolic/SymOp.hpp"
#include "vkcnn/common/symbolic/SymValue.hpp"
#include <cstdlib>
#include <cstring>
#include <fmt/base.h>
#include <tuple>

namespace vkcnn {

// NOTE: For SymGraphs we can make the following assumptions
// - All nodes have exactly one incomming hyperedge.
// - The amount of outgoing hyperedges is variable.
class SymConstGraph {
public:
  struct Header {
    std::size_t paramCount;
    std::size_t exprCount;
    std::size_t byteSize;
  };

  struct ConstantRecord {
    std::uint64_t value;
  };

  struct IdOrConstant {
    std::uint64_t isConstant : 1;
    std::uint64_t value : 63;
  };

  struct SymExprRecord {
    SymExpr expr; // <- expression which produces this node.
    IdOrConstant param0;
    IdOrConstant param1;
  };

  static std::tuple<SymConstGraph, SymIdMapping>
  compile(const hypergraph::ConstGraph<SymValue, SymExpr> &graph) {

    std::size_t nodeCount = graph.nodeCount();
    std::size_t constantCount = 0;
    std::size_t paramCount = 0;

    std::vector<hypergraph::NodeId> topoOrder =
        hypergraph::topologicalSort(graph);
    assert(topoOrder.size() == nodeCount);

    for (std::size_t nx = 0; nx < nodeCount; ++nx) {
      hypergraph::NodeId nid = topoOrder[nx];

      auto symValue = graph.get(nid);
      if (symValue.isConstant()) {
        constantCount += 1;
      } else if (graph.incoming(nid).size() == 0) {
        paramCount += 1;
      } else {
        // NOTE: Because we iterate in topo order, we know that all following
        // nodes are intermediate values.
        break;
      }
    }
    std::size_t exprCount = nodeCount - (constantCount + paramCount);

    std::vector<hypergraph::NodeId> idMapping(nodeCount, hypergraph::NodeId{0});

    std::size_t paramId = 0;

    for (std::size_t nx = 0; nx < nodeCount; ++nx) {
      std::size_t nxx = topoOrder[nx];
      if (nx < (constantCount + paramCount)) {
        std::size_t nxx = topoOrder[nx];
        hypergraph::NodeId n{nxx};
        auto v = graph.get(n);
        if (v.isConstant()) {
          idMapping[nxx] = hypergraph::NodeId{hypergraph::NodeId::NullId};
        } else {
          // NOTE: for parameter ids the order in which they where added to a
          // AdjGraph matters! All convertions used so far are stable!
          idMapping[nxx] = hypergraph::NodeId{paramId++};
        }
      } else {
        idMapping[nxx] = hypergraph::NodeId{nx - constantCount};
      }
    }

    // TODO construct mapping (id -> id) while considering inlined constants.

    std::size_t headerOffset = 0;
    std::size_t varOffset =
        align_up<alignof(SymExprRecord)>(headerOffset + sizeof(Header));
    std::size_t byteSize = varOffset + sizeof(SymExprRecord) * exprCount;

    void *ptr = malloc(byteSize);
    new (ptr) Header{paramCount, exprCount, byteSize};

    auto varPtr = reinterpret_cast<SymExprRecord *>(
        static_cast<std::byte *>(ptr) + varOffset);

    for (std::size_t x = 0; x < exprCount; ++x) {
      std::size_t nx = topoOrder[x + constantCount + paramCount];
      hypergraph::NodeId n{nx};
      auto incoming = graph.incoming(n);
      assert(incoming.size() == 1 && "All intermediate values must only have a "
                                     "single expr which produces them");
      auto exprid = incoming[0];
      SymExpr expr = graph.get(exprid);
      auto srcs = graph.src(exprid);
      assert(srcs.size() == 2 && "All expressions must be binary expressions");

      hypergraph::NodeId param0Id = srcs[0];
      auto param0Value = graph.get(param0Id);
      IdOrConstant param0;
      if (param0Value.isConstant()) {
        param0.isConstant = true;
        param0.value = param0Value.value();
      } else {
        param0.isConstant = false;
        std::uint64_t id = static_cast<std::uint64_t>(idMapping[param0Id]);
        assert(id != hypergraph::NodeId::NullId);
        param0.value = id;
      }

      hypergraph::NodeId param1Id = srcs[1];
      auto param1Value = graph.get(param1Id);
      IdOrConstant param1;
      if (param1Value.isConstant()) {
        param1.isConstant = true;
        param1.value = param1Value.value();
      } else {
        param1.isConstant = false;
        std::uint64_t id = static_cast<std::uint64_t>(idMapping[param1Id]);
        assert(id != hypergraph::NodeId::NullId);
        param1.value = id;
      }

      new (varPtr++) SymExprRecord{expr, param0, param1};
    }

    return std::make_tuple(SymGraph{ptr}, SymIdMapping{std::move(idMapping)});
  }

  std::size_t paramCount() const {
    return static_cast<const Header *>(m_buffer)->paramCount;
  }

  std::size_t varCount() const {
    return static_cast<const Header *>(m_buffer)->exprCount + paramCount();
  }

  std::size_t byteSize() const {
    return static_cast<const Header *>(m_buffer)->byteSize;
  }

  std::span<const SymExprRecord> expressions() const {
    auto raw = align_up<alignof(SymExprRecord)>(
        static_cast<const std::byte *>(m_buffer) + sizeof(Header));
    return std::span{reinterpret_cast<const SymExprRecord *>(raw), varCount()};
  }

  SymEval eval(std::span<std::uint64_t> params) {
    assert(params.size() == paramCount());
    std::vector<std::uint64_t> dp(varCount(), 0);
    std::memcpy(dp.data(), params.data(), params.size_bytes());

    auto exprs = expressions();
    for (std::size_t x = params.size(), e = 0; x < dp.size(); ++x, ++e) {
      const SymExprRecord &expr = exprs[e];
      std::uint64_t a;
      if (expr.param0.isConstant) {
        a = expr.param0.value;
      } else {
        a = dp[expr.param0.value];
      }
      std::uint64_t b;
      if (expr.param1.isConstant) {
        b = expr.param1.value;
      } else {
        b = dp[expr.param1.value];
      }
      std::uint64_t v;
      switch (expr.expr) {
      case SymExpr::CeilDiv:
        v = (a + b - 1) / b;
        break;
      case SymExpr::FloorDiv:
        v = a / b;
        break;
      case SymExpr::AlignUp:
        v = ((a + b - 1) / b) * b;
        break;
      case SymExpr::Mod:
        v = a % b;
        break;
      case SymExpr::Sub:
        v = a - b;
        break;
      case SymExpr::Mul:
        v = a * b;
        break;
      case SymExpr::Add:
        v = a + b;
        break;
      case SymExpr::Max:
        v = std::max(a, b);
        break;
      case SymExpr::Min:
        v = std::min(a, b);
        break;
      }
      dp[x] = v;
    }

    return SymEval{std::move(dp)};
  }

  SymEval eval(std::uint64_t param0) { return eval(std::span{&param0, 1}); }

  SymEval eval(std::uint64_t param0, std::uint64_t param1) {
    std::uint64_t params[2];
    params[0] = param0;
    params[1] = param1;
    return eval(params);
  }

private:
  SymConstGraph(void *buffer) : m_buffer(buffer) {}
  void *m_buffer;
};

} // namespace vkcnn

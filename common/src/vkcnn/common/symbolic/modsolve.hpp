#pragma once
#include <cstdint>
#include <optional>

#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/symbolic/SymOp.hpp"
#include "vkcnn/common/symbolic/SymValue.hpp"

namespace vkcnn::sym {

std::optional<std::uint64_t>
modsolve(const hypergraph::ConstGraph<SymValue, SymExpr> &graph,
         hypergraph::NodeId target, std::uint64_t mod);

} // namespace vkcnn::sym

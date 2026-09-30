#pragma once

#include "denox/compiler/implement/SuperGraphEdge.hpp"
#include "denox/compiler/implement/TensorId.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/ConstGraph.hpp"
#include "denox/memory/hypergraph/EdgeId.hpp"
#include <chrono>

namespace denox::compiler::selection {

memory::vector<memory::EdgeId> cache_aware_dispatch_order(
    const memory::ConstGraph<TensorId, SuperGraphEdge,
                             std::chrono::duration<float, std::milli>>
        &minCostSubgraph);

} // namespace denox::compiler::selection

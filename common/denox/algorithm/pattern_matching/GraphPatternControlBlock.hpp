#pragma once

#include "denox/algorithm/pattern_matching/EdgePattern.fwd.hpp"
#include "denox/algorithm/pattern_matching/NodePattern.fwd.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"

namespace denox::algorithm::pattern_matching::details {

template <typename V, typename E, typename W = memory::NullWeight>
struct GraphPatternControlBlock {
  memory::vector<denox::algorithm::NodePatternHandle<V, E, W>> nodes;
  memory::vector<denox::algorithm::EdgePatternHandle<V, E, W>> edges;
  std::size_t nextNodePatternId = 0;
  std::size_t nextEdgePatternId = 0;

  denox::algorithm::NodePatternHandle<V, E, W> matchNode() {
    std::size_t id = nextNodePatternId++;

    denox::algorithm::NodePatternHandle<V, E, W> nodePattern =
        std::make_shared<denox::algorithm::NodePattern<V, E, W>>(this, id);
    nodes.push_back(nodePattern);
    return nodePattern;
  }

  denox::algorithm::EdgePatternHandle<V, E, W> matchEdge() {
    std::size_t id = nextEdgePatternId++;
    denox::algorithm::EdgePatternHandle<V, E, W> edgePattern =
        std::make_shared<denox::algorithm::EdgePattern<V, E, W>>(this, id);
    edges.push_back(edgePattern);
    return edgePattern;
  }
};

} // namespace denox::algorithm::pattern_matching::details

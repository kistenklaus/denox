#pragma once

#include "denox/memory/allocator/mallocator.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/hypergraph/LinkedGraph.hpp"
#include "denox/memory/hypergraph/NodeId.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"
namespace denox::algorithm {

template<typename V, typename E, typename W = memory::NullWeight, typename Allocator = memory::mallocator>
std::size_t count_children(const typename memory::LinkedGraph<V, E, W, Allocator>::NodeHandle& node) {
  using Graph = memory::LinkedGraph<V, E, W, Allocator>;
  using NodeHandle = Graph::NodeHandle;
  
  memory::vector<NodeHandle> stack;
  stack.reserve(node.upperNodeCount());
  memory::dynamic_bitset visited(node.upperNodeCount());

  stack.push_back(node);

  std::size_t nodeCount = 0;
  while(!stack.empty()) {
    NodeHandle node =stack.back();
    stack.pop_back();
    memory::NodeId nid = node->id();
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;
    ++nodeCount;
    for (const auto& e : node->outgoing()) {
      stack.emplace_back(e.dst());
    }
  }
  return nodeCount;
}

}

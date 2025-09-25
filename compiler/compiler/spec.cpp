#include "compiler/spec.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/NodeId.hpp"
#include "memory/hypergraph/NullWeight.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include <exception>

namespace denox::compiler {

using Graph = LinkedModel::Graph;
using NodeHandle = Graph::NodeHandle;

static void ensure_specialized_type(NodeHandle &node) {
  if (node->value().type().has_value()) {
    return;
  }
  node->value().setType(memory::Dtype::F16);
}

void specialize(LinkedModel &model,
                memory::span<const memory::ActivationLayout> layouts) {
  assert(!layouts.empty());
  if (layouts.empty()) {
    std::terminate();
  }

  memory::vector<NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());
  memory::dynamic_bitset visited(model.graph.upperNodeCount() * layouts.size());

  stack.push_back(model.input);

  assert(model.input->value().layout().has_value());
  assert(model.output->value().layout().has_value());

  while (!stack.empty()) {
    NodeHandle node = std::move(stack.back());
    stack.pop_back();
    memory::NodeId nid = node->id();
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;

    ensure_specialized_type(node);

    if (node->value().layout().has_value()) {
      // simple recurse.
      for (const auto &op : node->outgoing()) {
        stack.emplace_back(op.dst());
      }
    } else {
      // if layout is not specialized.
      auto it = layouts.begin();
      node->value().setLayout(*it);

      for (const auto &op : node->outgoing()) {
        stack.emplace_back(op.dst());
      }
      while (++it != layouts.end()) {
        if (!it->supports(node->value().channels())) {
          continue;
        }
        auto snode = model.graph.createNode(node->value());
        assert(snode->id() < visited.size());
        visited[snode->id()] = false;
        // fmt::println("Created node {} as instance of {}",
        // static_cast<std::uint64_t>(snode->id()),
        // static_cast<std::uint64_t>(nid));
        snode->value().setLayout(*it);
        auto incoming = snode->incoming();
        for (const auto &edge : node->incoming()) {
          memory::small_vector<NodeHandle, 2> srcs;
          memory::small_vector<const NodeHandle *, 2> srcsp;
          for (const auto &s : edge.srcs()) {
            srcs.emplace_back(s);
          }
          for (const auto &h : srcs) {
            srcsp.push_back(&h);
          }
          incoming.insert_after_with_dynamic_srcs(
              incoming.begin(), memory::span<const NodeHandle *>(srcsp),
              memory::NullWeight{}, edge.value());
        }

        for (const auto &edge : node->outgoing()) {
          memory::small_vector<NodeHandle, 1> additionalSrcs;
          memory::small_vector<const NodeHandle *, 1> additionalSrcsp;

          for (const auto &s : edge.srcs()) {
            if (&s == node.operator->()) {
              continue;
            }
            additionalSrcs.emplace_back(s);
          }

          for (const auto &h : additionalSrcs) {
            additionalSrcsp.push_back(&h);
          }

          NodeHandle dst{edge.dst()};

          snode->outgoing().insert_after_with_dynamic_srcs(
              snode->outgoing().begin(), additionalSrcsp, dst,
              memory::NullWeight{}, edge.value());
        }
        stack.push_back(snode);
      }
    }
  }
}

} // namespace denox::compiler

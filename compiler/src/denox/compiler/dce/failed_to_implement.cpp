#include "denox/compiler/dce/failed_to_implement.hpp"
#include "denox/algorithm/minimum_const_subgraph.hpp"
#include "denox/compiler/dce/ConstModel.hpp"
#include <absl/strings/internal/str_format/extension.h>
#include <stdexcept>

void denox::compiler::failed_to_implement(const SuperGraph &supergraph,
                                          const ConstModel &model) {

  // ===== Explaination of the algorithmic idea ========
  // Problem:
  // Given a graph G with nodes V and edges E and additional edges E'.
  // Find the minimum amount of edges from E' to add to E, such that there
  // exists a hyperpath from all inputs nodes to all output nodes.
  //
  // Algorithm: Define graph G=(V, E & E')
  // With cost function c(e) = 0, if e \in E otherwise c(e) = 1.
  // Then find minimum-cost-subgraph of G.
  // All edges of the minimum-cost-subgraph, which have cost c(e) = 1, are
  // part of the minimum edge set from E' to add to E such that there exists
  // a path.
  //
  // Mapping: E are edges of the supergraph and E' are edges of the model.
  // Both supergraph and model have the same nodes, therefor this works and in
  // my opinion is beautifully simple. Although probably not optimal for error
  // message this is more than enough!

  assert(supergraph.graph.nodeCount() == model.graph.nodeCount());
  const size_t N = supergraph.graph.nodeCount();

  struct Phantom {};

  memory::AdjGraph<Phantom, memory::EdgeId, uint32_t> agraph;
  for (uint32_t n = 0; n < N; ++n) {
    agraph.addNode({});
  }

  uint32_t M = static_cast<uint32_t>(supergraph.graph.edgeCount());
  for (uint32_t e = 0; e < M; ++e) {
    memory::EdgeId eid{e};
    agraph.addEdge(supergraph.graph.src(eid), supergraph.graph.dst(eid),
                   memory::EdgeId{}, 0);
  }

  uint32_t K = static_cast<uint32_t>(model.graph.edgeCount());
  for (uint32_t e = 0; e < K; ++e) {
    memory::EdgeId eid{e};
    agraph.addEdge(model.graph.src(eid), model.graph.dst(eid), eid, 1);
  }

  memory::ConstGraph<Phantom, memory::EdgeId, uint32_t> graph{
      std::move(agraph)};
  memory::AdjGraph<Phantom, memory::EdgeId, uint32_t> aminimum_cost_subgraph =
      algorithm::minimum_cost_subgraph(graph, supergraph.inputs,
                                       supergraph.outputs);
  memory::ConstGraph<Phantom, memory::EdgeId, uint32_t> minimum_cost_subgraph{
      std::move(aminimum_cost_subgraph)};

  memory::vector<memory::EdgeId> unimplemented_ops;

  for (uint32_t e = 0; e < minimum_cost_subgraph.edgeCount(); ++e) {
    memory::EdgeId eid{e};
    if (minimum_cost_subgraph.weight(eid) == 1) {
      memory::EdgeId opid = minimum_cost_subgraph.get(eid);
      assert(opid != memory::EdgeId{});
      unimplemented_ops.push_back(opid);
    }
  }

  std::ranges::reverse(unimplemented_ops);

  // TODO proper error message!
  std::string msg;
  for (memory::EdgeId eid : unimplemented_ops) {
    const ComputeOp &op = model.graph.get(eid);
    const auto srcsIds = model.graph.src(eid);
    const auto dstId = model.graph.dst(eid);

    std::string srcString;
    if (srcsIds.size() == 1) {
      const TensorInstance &tensor = model.graph.get(srcsIds.front());
      srcString =
          fmt::format("{}[{}:{}]", tensor.format, tensor.channels, tensor.type);
    } else {
      srcString = "{";
      bool first = true;
      for (memory::NodeId src : srcsIds) {
        const TensorInstance &tensor = model.graph.get(src);
        if (!first) {
          srcString += ",";
        }
        first = false;
        srcString += fmt::format("{}[{}:{}]", tensor.format, tensor.channels,
                                 tensor.type);
      }
      srcString += "}";
    }
    const auto &dstTensor = model.graph.get(dstId);
    std::string dstString = fmt::format("{}[{}:{}]", dstTensor.format,
                                        dstTensor.channels, dstTensor.type);

    std::string opString;
    std::string with;
    switch (op.tag()) {
    case ComputeOpKind::None:
      opString = "noop";
      break;
    case ComputeOpKind::Conv:
      opString = fmt::format("conv{}x{}", op.conv()->W->shape().r,
                             op.conv()->W->shape().s);
      with = fmt::format("{{padding=({},{}), stride=({},{})}}",
                         op.conv()->padding.x, op.conv()->padding.y,
                         op.conv()->stride.x, op.conv()->stride.y);
      break;
    case ComputeOpKind::Activation: {
      switch (op.activation().func.kind()) {
      case ActivationFunctionKind::ReLU:
        opString = "relu";
        break;
      case ActivationFunctionKind::LeakyReLU:
        opString = "leaky-relu";
        break;
      case ActivationFunctionKind::SiLU:
        opString = "silu";
        break;
      case ActivationFunctionKind::Swish:
        opString = "swish";
        break;
      }
      break;
    }
    case ComputeOpKind::Upsample:
      switch (op.upsample().mode) {
      case FilterMode::Nearest:
        opString = "nearest-upsample";
        with =
            fmt::format("{{scaling_factor={}}}", op.upsample().scalingFactor);
        break;
      }
      break;
    case ComputeOpKind::Pool:
      switch (op.pool()->func) {
      case PoolFunction::Max:
        opString = fmt::format("max-pool{}x{}", op.pool()->kernelSize.x,
                               op.pool()->kernelSize.y);
        with = fmt::format("{{padding=({},{}), stride=({},{})}}",
                           op.pool()->padding.x, op.pool()->padding.y,
                           op.pool()->stride.x, op.pool()->stride.y);
        break;
      case PoolFunction::Avg:
        opString = fmt::format("avg-pool{}x{}", op.pool()->kernelSize.x,
                               op.pool()->kernelSize.y);
        with = fmt::format("{{padding=({},{}), stride=({},{})}}",
                           op.pool()->padding.x, op.pool()->padding.y,
                           op.pool()->stride.x, op.pool()->stride.y);
        break;
      }
      break;
    case ComputeOpKind::Concat:
      opString = "channel-concat";
      break;
    case ComputeOpKind::Pad:
      opString = "pad";
      break;
    case ComputeOpKind::Slice:
      opString = "slice";
      break;
    }

    msg += fmt::format("{:>25} {:-^50} {}\n", srcString, opString, dstString);
    if (!with.empty()) {
      msg += fmt::format("{:>25} {:^50}\n", "", fmt::format("with: {}", with));
    }
  }
  msg.pop_back(); // pop last '\n' line break

  throw std::runtime_error(fmt::format(
      "Failed to implement at least one of the following operations:\n{}",
      msg));
}

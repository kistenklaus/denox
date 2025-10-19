#include "diag/failed_to_realize.hpp"
#include "compiler/impl/ComputeOpImpl.hpp"
#include "diag/invalid_state.hpp"
#include "diag/unreachable.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include <stdexcept>

using ComputeOpImpl = denox::compiler::impl::details::ComputeOpImpl;

namespace denox::compiler::diag {

void failed_to_realize(const OpModel &opModel,
                       const memory::ConstGraph<TensorInstance, ComputeOpImpl,
                                                float> &supergraph) {
  memory::vector<memory::NodeId> stack;
  memory::dynamic_bitset visited(opModel.graph.nodeCount(), false);
  // NOTE: opModel.graph.nodeCount is a upper limit
  // and not the exact amount of values!
  memory::dynamic_bitset valuesReachable(opModel.graph.nodeCount(), false);
  stack.push_back(opModel.input);

  while (!stack.empty()) {
    memory::NodeId nid = stack.back();
    stack.pop_back();
    auto tensor = supergraph.get(nid);
    valuesReachable[tensor.valueId()] = true;

    if (visited[*nid]) {
      continue;
    }
    visited[*nid] = true;

    auto outgoing = supergraph.outgoing(nid);
    for (memory::EdgeId eid : outgoing) {
      memory::NodeId dst = supergraph.dst(eid);
      stack.push_back(dst);
    }
  }
  assert(!visited[*opModel.output]);
  // Find first value that was not reached
  std::uint64_t v = 0;
  for (v = 0; v < opModel.graph.nodeCount(); ++v) {
    if (!valuesReachable[v]) {
      break;
    }
  }
  memory::vector<memory::NodeId> notreached;
  for (std::uint64_t n = 0; n < opModel.graph.nodeCount(); ++n) {
    memory::NodeId nid{n};
    auto tensor = opModel.graph.get(nid);
    if (tensor.valueId() == v) {
      notreached.push_back(nid);
    }
  }
  std::string msg =
      "Failed to realize model:\n"
      "One of the following operation could not be implemented:\n";

  for (memory::NodeId dst : notreached) {

    auto dstNode = opModel.graph.get(dst);
    std::string dstStr =
        fmt::format("{}[{}:{}]", dstNode.layout.to_string(), dstNode.channels,
                    dstNode.type.to_string());

    auto incoming = opModel.graph.incoming(dst);
    for (memory::EdgeId e : incoming) {
      auto op = opModel.graph.get(e);
      auto srcs = opModel.graph.src(e);
      bool first = true;
      std::string srcStr = " ";
      for (memory::NodeId src : srcs) {
        auto node = opModel.graph.get(src);
        if (!first) {
          srcStr += ",";
        }
        first = false;
        srcStr += fmt::format("{}[{}:{}]", node.layout.to_string(),
                              node.channels, node.type.to_string());
      }
      std::string opString;
      std::string params;
      switch (op.tag()) {
      case ComputeOpTag::None:
        compiler::diag::invalid_state();
      case ComputeOpTag::Conv: {
        auto conv = op.conv();
        opString +=
            fmt::format("Conv{}x{}", conv->W->shape().s, conv->W->shape().r);
        params += fmt::format(
            "with: padding=({},{}), stride=({},{}), bias={}, atype={}",
            conv->padding.x, conv->padding.y, conv->stride.x, conv->stride.y,
            conv->B != nullptr,
            conv->atype.has_value() ?  conv->atype->to_string() : "none");
        break;
      }
      case ComputeOpTag::Activation:
        opString += fmt::format("activation");
        break;
      case ComputeOpTag::Upsample:
        opString += fmt::format("upsample");
        break;
      case ComputeOpTag::Pool:
        opString += fmt::format("pool");
        break;
      case ComputeOpTag::Concat:
        opString += fmt::format("concat");
        break;
      case ComputeOpTag::Pad:
        opString += fmt::format("pad");
        break;
      case ComputeOpTag::Slice:
        opString += fmt::format("slice");
        break;
      }

      // fmt::println("{:>22} \x1B[34m{:-^40}>\x1B[0m {:<22} : {}", inStr,
      //              impl.shader->name(o.pattern),
      //              fmt::format("{}[{}]", dst.layout.to_string(),
      //              dst.channels),
      //              heuristic->weight_to_string(constSupergraph.weight(oid)));

      msg += fmt::format(" \u2022 {:-<22}{:-^40}> {:<20}\n", srcStr, opString,
                         dstStr);
      if (!params.empty()) {
        msg += fmt::format("      {}\n", params);
      }
    }
  }

  throw std::runtime_error(msg);
}

} // namespace denox::compiler::diag

#include "./debug.hpp"
#include <fmt/base.h>
#include <stdexcept>

void vkcnn::debug::print_const_graph(
    const hypergraph::ConstGraph<ComputeTensor, ComputeOp>& graph) {

  for (std::size_t n = 0; n < graph.nodeCount(); ++n) {
    hypergraph::NodeId nodeId{n};
    const ComputeTensor& tensor = graph.get(nodeId);
    std::string dtypename;
    if (tensor.type().has_value()) {
      if (*tensor.type() == FloatType::F16) {
        dtypename = "f16";
      } else if (*tensor.type() == FloatType::F32) {
        dtypename = "f32";
      } else if (*tensor.type() == FloatType::F64) {
        dtypename = "f64";
      } else {
        throw std::logic_error("unexpected");
      }
    } else {
      dtypename = "any";
    }
    
    fmt::println("[{}]: {{type: {}}}", n, dtypename);
  }

  for (std::size_t e = 0; e < graph.edgeCount(); ++e) {
    hypergraph::EdgeId edgeId{e};
    const ComputeOp& op = graph.get(edgeId);
    auto srcs = graph.src(edgeId);
    auto dst = graph.dst(edgeId);
  }

}

void vkcnn::debug::print_const_graph(
    const hypergraph::ConstGraph<compiler::SpecializedTensor, ComputeOp>& graph) {}


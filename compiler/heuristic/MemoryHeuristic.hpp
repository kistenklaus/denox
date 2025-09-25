#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "heuristic/IHeuristic.hpp"
#include "memory/container/vector.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/IShader.hpp"
#include "symbolic/SymGraph.hpp"
namespace denox::compiler {

class MemoryHeuristic : public IHeuristic {
public:
  explicit MemoryHeuristic(
      std::span<const IShader *> shaders,
      const memory::ConstGraph<ComputeTensor, ComputeOp> *opGraph,
      const SymGraph &symGraph, memory::NodeId inputId)
      : m_shaders(shaders), m_opGraph(opGraph) {
    // evaluate all symbolic values for 1920x1080
    const ComputeTensor &input = opGraph->get(inputId);
    // TODO requires Symbolic engine evaluation
    const sym_vec2 inputExtent = input.extent();
  }

  float eval(std::span<const ComputeTensor *> ins, const ComputeTensor &out,
            unsigned int pattern,
            const algorithm::ConstGraphMatch<ComputeTensor, ComputeOp> &match,
            unsigned int shaderId) const final override {
    return 1.0f;
    const IShader *shader = m_shaders[shaderId];

    std::size_t byteSize =
        shader->parameterMemorySize(*m_opGraph, pattern, match);
    for (const ComputeTensor *in : ins) {
      sym_vec2 extent = in->extent();
      Sym::value_type W;
      if (extent.x.isConstant()) {
        W = extent.x.constant();
      } else {
        W = m_symGraphEval[extent.x.symbol()];
      }
      Sym::value_type H;
      if (extent.y.isConstant()) {
        H = extent.y.constant();
      } else {
        H = m_symGraphEval[extent.y.symbol()];
      }
      byteSize += static_cast<std::size_t>(W) * static_cast<std::size_t>(H) *
                  in->type()->size();
    }
    sym_vec2 extent = out.extent();
    Sym::value_type W;
    if (extent.x.isConstant()) {
      W = extent.x.constant();
    } else {
      W = m_symGraphEval[extent.x.symbol()];
    }
    Sym::value_type H;
    if (extent.y.isConstant()) {
      H = extent.y.constant();
    } else {
      H = m_symGraphEval[extent.y.symbol()];
    }
    byteSize += static_cast<std::size_t>(W) * static_cast<std::size_t>(H) *
                out.type()->size();
    return static_cast<float>(static_cast<double>(byteSize) * 1e-6);
  }

private:
  std::span<const IShader *> m_shaders;
  const memory::ConstGraph<ComputeTensor, ComputeOp> *m_opGraph;
  memory::vector<Sym::value_type> m_symGraphEval;
};

} // namespace denox::compiler

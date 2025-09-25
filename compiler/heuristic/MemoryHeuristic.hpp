#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "compiler/ir/SpecModel.hpp"
#include "heuristic/IHeuristic.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "shaders/IShader.hpp"
#include "symbolic/SymGraph.hpp"
#include "symbolic/SymGraphEval.hpp"
namespace denox::compiler {

class MemoryHeuristic : public IHeuristic {
public:
  explicit MemoryHeuristic(
      std::span<const IShader *> shaders,
      const memory::ConstGraph<TensorInstance, ComputeOp> *opGraph,
      const SymGraph &symGraph, memory::NodeId inputId)
      : m_shaders(shaders), m_opGraph(opGraph) {
    // evaluate all symbolic values for 1920x1080
    const TensorInstance &input = opGraph->get(inputId);
    // TODO requires Symbolic engine evaluation
    const sym_vec2 inputExtent = input.extent;
    memory::small_vector<SymSpec, 2> symSpecs;
    if (inputExtent.x.isSymbolic()) {
      symSpecs.emplace_back(inputExtent.x.symbol(), Sym::value_type(1920));
    }
    if (inputExtent.y.isSymbolic()) {
      symSpecs.emplace_back(inputExtent.y.symbol(), Sym::value_type(1080));
    }
    m_eval = symGraph.eval(symSpecs);
  }

  float eval(std::span<const TensorInstance *> ins, const TensorInstance &out,
             unsigned int pattern,
             const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
             const IShader *shader) const final override {

    std::size_t byteSize =
        shader->parameterMemorySize(*m_opGraph, pattern, match);
    for (const TensorInstance *in : ins) {
      sym_vec2 extent = in->extent;
      Sym::value_type W = *m_eval[extent.x];
      Sym::value_type H = *m_eval[extent.y];
      std::size_t n = static_cast<std::size_t>(W) *
                      static_cast<std::size_t>(H) * in->type.size();
      if (in->layout.isVectorized()) {
        byteSize += (n * 9) / 10;
      } else {
        byteSize += n;
      }
    }
    sym_vec2 extent = out.extent;
    Sym::value_type W = *m_eval[extent.x];
    Sym::value_type H = *m_eval[extent.y];
    std::size_t n = static_cast<std::size_t>(W) * static_cast<std::size_t>(H) *
                    out.type.size();
    if (out.layout.isVectorized()) {
      byteSize += (n * 9) / 10;
    } else {
      byteSize += n;
    }
    return static_cast<float>(static_cast<double>(byteSize) * 1e-6);
  }

private:
  std::span<const IShader *> m_shaders;
  const memory::ConstGraph<TensorInstance, ComputeOp> *m_opGraph;
  SymGraphEval m_eval;
};

} // namespace denox::compiler

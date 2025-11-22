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
    const TensorInstance &input = opGraph->get(inputId);
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
             unsigned int config,
             const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
             const IShader *shader) const final override {

    std::size_t byteSize =
        shader->parameterMemorySize(*m_opGraph, pattern, match);
    for (const TensorInstance *in : ins) {
      sym_vec2 extent = in->extent;
      Sym::value_type W = *m_eval[extent.x];
      Sym::value_type H = *m_eval[extent.y];
      std::size_t n = static_cast<std::size_t>(W) *
                      static_cast<std::size_t>(H) * in->type.size() *
                      in->channels;
      byteSize += n;
    }
    sym_vec2 extent = out.extent;
    Sym::value_type W = *m_eval[extent.x];
    Sym::value_type H = *m_eval[extent.y];
    std::size_t n = static_cast<std::size_t>(W) * static_cast<std::size_t>(H) *
                    out.type.size() * out.channels;
    byteSize += n;

    return static_cast<float>(static_cast<double>(byteSize) * 1e-6 *
                              static_cast<double>(shader->speedup(config)));
  }

  memory::string weight_to_string(float weight) const final override {
    if (weight > 1000.0f) {
      return fmt::format("\x1B[1m{:.3f}GB\x1B[0m", weight * 1e-3f);
    } else if (weight >= 1.0f) {
      return fmt::format("\x1B[1m{:.3f}MB\x1B[0m", weight);
    } else if (weight >= 1e-3f) {
      // KB range
      return fmt::format("{:.3f}KB", weight * 1e3f);
    } else if (weight == 0.0f) {
      return "0B";
    } else {
      // B range.
      return fmt::format("{:.3f}B", weight * 1e6f);
    }
  }

private:
  std::span<const IShader *> m_shaders;
  const memory::ConstGraph<TensorInstance, ComputeOp> *m_opGraph;
  SymGraphEval m_eval;
};

} // namespace denox::compiler

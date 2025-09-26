#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "compiler/ir/SpecModel.hpp"
#include "memory/container/string.hpp"
#include "memory/hypergraph/ConstGraph.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"

namespace denox::compiler {

struct ShaderOp {
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;
  Pattern pattern;
  memory::small_vector<Pattern::NP, 1> inputs;
  Pattern::NP output;

  ShaderOp(Pattern pattern, Pattern::NP input, Pattern::NP output)
      : pattern(std::move(pattern)), inputs{std::move(input)},
        output(std::move(output)) {}

  ShaderOp(Pattern pattern, Pattern::NP in0, Pattern::NP in1,
           Pattern::NP output)
      : pattern(std::move(pattern)), inputs{std::move(in0), std::move(in1)},
        output(std::move(output)) {}
};

struct ShaderCapabilities {
  memory::vector<ShaderOp> patterns;
};

class IShader {
public:
  virtual ~IShader() = default;

  virtual const ShaderCapabilities &capabilities() const = 0;

  virtual memory::optional<unsigned int> acceptMatch(
      [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
          &graph,
      unsigned int pattern,
      [[maybe_unused]] const algorithm::ConstGraphMatch<
          TensorInstance, ComputeOp> &match) const {
    return pattern;
  }

  virtual std::size_t parameterMemorySize(
      [[maybe_unused]] const memory::ConstGraph<TensorInstance, ComputeOp>
          &graph,
      [[maybe_unused]] unsigned int pattern,
      [[maybe_unused]] const algorithm::ConstGraphMatch<
          TensorInstance, ComputeOp> &match) const {
    return 0;
  }

  virtual float speedup([[maybe_unused]] unsigned int pattern) const {
    return 1.0f;
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  virtual void
  implement(unsigned int pattern,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match)
      const = 0;

  virtual memory::string name(unsigned int pattern) const = 0;
};

} // namespace denox::compiler

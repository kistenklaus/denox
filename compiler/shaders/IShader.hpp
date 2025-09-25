#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"

namespace denox::compiler {

struct ShaderOp {
  algorithm::GraphPattern<ComputeTensor, ComputeOp> pattern;
  algorithm::NodePatternHandle<ComputeTensor, ComputeOp> input;
  algorithm::NodePatternHandle<ComputeTensor, ComputeOp> output;
};

struct ShaderCapabilities {
  memory::vector<ShaderOp> patterns;
};

class IShader {
public:
  virtual ~IShader() = default;

  virtual const ShaderCapabilities &capabilities() const = 0;

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  virtual void
  implement(unsigned int pattern,
            const algorithm::ConstGraphMatch<ComputeTensor, ComputeOp> &match)
      const = 0;
};

} // namespace denox::compiler

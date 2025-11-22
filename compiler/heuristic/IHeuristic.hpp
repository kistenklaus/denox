#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include "shaders/IShader.hpp"
#include <span>
namespace denox::compiler {

class IHeuristic {
public:
  virtual ~IHeuristic() = default;

  virtual float
  eval(std::span<const TensorInstance *> in, const TensorInstance &out,
            unsigned int pattern,
            unsigned int config,
            const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
            const IShader* shader) const = 0;

  virtual memory::string weight_to_string(float weight) const {
    return fmt::format("{}", weight);
  }
};

} // namespace denox::compiler

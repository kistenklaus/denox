#pragma once

#include "algorithm/pattern_matching/ConstGraphMatch.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
#include <span>
namespace denox::compiler {

class IHeuristic {
public:
  virtual ~IHeuristic() = default;

  virtual float
  eval(std::span<const ComputeTensor *> in, const ComputeTensor &out,
            unsigned int pattern,
            const algorithm::ConstGraphMatch<ComputeTensor, ComputeOp> &match,
            unsigned int shader) const = 0;
};

} // namespace denox::compiler

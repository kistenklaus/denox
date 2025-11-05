#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "model/ComputeOp.hpp"
#include "model/ComputeTensor.hpp"
namespace denox::compiler::cano {

class IFusionRule {
public:
  virtual ~IFusionRule() = default;

  virtual const algorithm::GraphPattern<ComputeTensor, ComputeOp> &
  pattern() = 0;

  virtual void
  apply(SymGraph& symGraph, const algorithm::LinkedGraphMatch<ComputeTensor, ComputeOp> &match) = 0;
};

} // namespace denox::compiler::cano

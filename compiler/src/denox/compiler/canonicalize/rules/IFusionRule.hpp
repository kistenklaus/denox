#pragma once

#include "denox/algorithm/pattern_matching/GraphPattern.hpp"
#include "denox/algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "denox/common/TensorDescriptor.hpp"
#include "denox/compiler/frontend/model/ComputeOp.hpp"
#include "denox/compiler/frontend/model/ComputeTensor.hpp"
namespace denox::compiler::cano {

class IFusionRule {
public:
  virtual ~IFusionRule() = default;

  virtual const algorithm::GraphPattern<TensorDescriptor, ComputeOp> &
  pattern() = 0;

  virtual void
  apply(SymGraph& symGraph, const algorithm::LinkedGraphMatch<TensorDescriptor, ComputeOp> &match) = 0;
};

} // namespace denox::compiler::cano

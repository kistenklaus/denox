#pragma once

#include "vkcnn/common/compiler/CompileOptions.hpp"
#include "vkcnn/common/hypergraph/ConstGraph.hpp"
#include "vkcnn/common/hypergraph/NodeId.hpp"
#include "vkcnn/common/model/ComputeOp.hpp"
#include "vkcnn/common/model/Model.hpp"
#include "vkcnn/common/model/SymTensorExtent.hpp"
#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"

namespace vkcnn::compiler {

struct SpecializedTensor {
  SymTensorExtent extent;
  unsigned int channels;
  FloatType dtype;
  ActivationLayout layout;
  hypergraph::NodeId tensorId;
};

//
hypergraph::ConstGraph<SpecializedTensor, ComputeOp>
specialize_tensors(const Model &model, const CompileOptions &options);

} // namespace vkcnn::compiler

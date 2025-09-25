#pragma once

#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "compiler/ir/CanoModel.hpp"
#include "symbolic/sym_vec2.hpp"

namespace denox::compiler {

struct TensorInstance {
  sym_vec2 extent;
  unsigned int channels;
  memory::ActivationLayout layout;
  memory::Dtype type;
  CanoModel::Graph::NodeHandle originalNode;

  std::uint64_t valueId() const { return originalNode->id(); }
};

}

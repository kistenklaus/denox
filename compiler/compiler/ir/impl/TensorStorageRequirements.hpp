#pragma once

#include "memory/dtype/dtype.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "symbolic/Sym.hpp"
#include "symbolic/sym_vec2.hpp"
#include <memory>
namespace denox::compiler {

struct TensorMeta {
  sym_vec2 extent;
  unsigned int channels;
  memory::ActivationLayout layout;
  memory::Dtype type;
};

struct TensorStorageRequirements {
  Sym byteSize;
  unsigned int minAlignment;
  std::unique_ptr<TensorMeta> meta;
};

} // namespace denox::compiler

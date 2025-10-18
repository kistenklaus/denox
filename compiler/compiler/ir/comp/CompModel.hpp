#pragma once

#include "compiler/ir/impl/InputDesc.hpp"
#include "compiler/ir/impl/OutputDesc.hpp"
#include "compiler/ir/impl/PushConstant.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "symbolic/Sym.hpp"
#include <cstdint>

namespace denox::compiler {

struct TensorView {
  std::uint64_t buffer;
  Sym offset;
};

struct Buffer {
  Sym size;
  unsigned int alignment;
  memory::optional<std::uint64_t> initalizer;
};

struct ShaderSourceView {
  std::uint64_t offset;
  std::uint64_t size;
};

struct DescriptorBinding {
  std::uint32_t binding;
  AccessFlag access;
  std::uint64_t tensor;
};

struct DescriptorSetBinding {
  std::uint32_t set;
  memory::vector<DescriptorBinding> bindings;
};

struct Dispatch {
  ShaderSourceView src;
  memory::vector<DescriptorSetBinding> setBindings;
  memory::vector<PushConstant> pushConstants;
};

struct CompModel {
  SymGraph symGraph;

  memory::vector<TensorView> tensors;
  memory::vector<memory::optional<TensorMeta>> tensorInfo;

  memory::vector<Dispatch> dispatches;
  memory::vector<Buffer> buffers;

  memory::vector<std::byte> roData;

  memory::vector<InputDesc> inputs;
  memory::vector<OutputDesc> outputs;
};

} // namespace denox::compiler

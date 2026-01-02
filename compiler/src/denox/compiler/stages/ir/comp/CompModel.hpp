#pragma once

#include "compiler/ir/impl/InputDesc.hpp"
#include "compiler/ir/impl/OutputDesc.hpp"
#include "compiler/ir/impl/PushConstant.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
#include "shaders/compiler/ShaderBinary.hpp"
#include "denox/symbolic/Sym.hpp"
#include <cstdint>
#include <vector>

namespace denox::compiler {

struct TensorView {
  std::uint64_t buffer;
  Sym offset;
  Sym size;
};

struct Buffer {
  Sym size;
  unsigned int alignment;
};

struct TensorInitializers {
  std::uint32_t tensor;
  std::vector<std::byte> data;
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
  std::uint32_t binaryId;
  std::array<Sym, 3> workgroupCount;
  memory::vector<DescriptorSetBinding> setBindings;
  memory::vector<PushConstant> pushConstants;
  std::optional<std::string> debug_info;
  std::optional<std::string> name;
  std::optional<std::string> input_desc;
  std::optional<std::string> output_desc;
  std::optional<Sym> memory_reads;
  std::optional<Sym> memory_writes;
};

struct CompModel {
  SymGraph symGraph;

  memory::vector<TensorView> tensors;
  memory::vector<memory::optional<TensorMeta>> tensorInfo;

  memory::vector<Dispatch> dispatches;
  memory::vector<ShaderBinary> shaderBinaries;
  memory::vector<Buffer> buffers;

  memory::vector<InputDesc> inputs;
  memory::vector<OutputDesc> outputs;

  memory::vector<TensorInitalizer> initializers;
};

} // namespace denox::compiler

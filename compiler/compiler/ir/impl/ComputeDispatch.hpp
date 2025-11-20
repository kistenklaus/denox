#pragma once

#include "compiler/ir/impl/PushConstant.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include <memory>

namespace denox::compiler {

struct ComputeDispatchMeta {
  memory::optional<memory::string> name;
  memory::optional<memory::string> debug_info;
  memory::optional<io::Path> sourcePath;
  memory::optional<memory::string> input_desc;
  memory::optional<memory::string> output_desc;
  memory::optional<Sym> memory_reads;
  memory::optional<Sym> memory_writes;
};

struct ComputeDispatch {
  std::array<Sym, 3> workgroupCount;
  std::uint32_t binaryId;
  memory::vector<TensorBinding> bindings;
  memory::vector<PushConstant> pushConstants;
  std::unique_ptr<ComputeDispatchMeta> meta;
};

} // namespace denox::compiler

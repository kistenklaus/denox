#pragma once

#include "denox/compiler/implement/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/vector.hpp"
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

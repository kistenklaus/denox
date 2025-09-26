#pragma once

#include "compiler/ir/impl/PushConstant.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include <cstdint>
#include <memory>

namespace denox::compiler {

struct ComputeDispatchMeta {
  memory::optional<memory::string> name;
  memory::optional<io::Path> sourcePath;
};

struct ComputeDispatch {
  memory::vector<std::uint32_t> spirvSrc;
  memory::vector<TensorBinding> bindings;
  memory::vector<PushConstant> pushConstants;
  std::unique_ptr<ComputeDispatchMeta> meta;
};

} // namespace denox::compiler

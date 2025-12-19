#pragma once

#include "compiler/ir/impl/TensorId.hpp"
#include "denox/memory/container/vector.hpp"
#include <cstddef>

namespace denox::compiler {

struct Parameter {
  TensorId tensorId;
  memory::vector<std::byte> data;
};

} // namespace denox::compiler

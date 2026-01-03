#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/db/Db.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/span.hpp"

namespace denox {

memory::vector<std::byte> compile(memory::span<const std::byte> onnx,
                                  memory::optional<Db> db,
                                  const compiler::CompileOptions &options);

} // namespace denox

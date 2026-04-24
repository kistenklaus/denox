#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/db/Db.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/runtime/context.hpp"

namespace denox {

void populate(Db db, memory::span<const std::byte> onnx,
              const compiler::CompileOptions &options, const diag::Logger& logger);

}

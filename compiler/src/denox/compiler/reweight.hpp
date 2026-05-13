#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/db/Db.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/runtime/context.hpp"

namespace denox {

memory::vector<std::byte> reweight(memory::span<const std::byte> dnx,
                                   memory::span<const std::byte> onnx,
                                   const diag::Logger &logger);
}

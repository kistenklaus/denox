#pragma once

#include "denox/diag/logging.hpp"
#include "denox/memory/container/span.hpp"

namespace denox {

void reweight(memory::span<std::byte> dnx, memory::span<const std::byte> onnx,
              const diag::Logger &logger);
}

#pragma once

#include "Options.hpp"
#include "memory/container/span.hpp"
namespace denox::compiler {

void entry(memory::span<const std::byte> raw, const Options &options);
}


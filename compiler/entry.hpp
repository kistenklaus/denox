#pragma once

#include "Options.hpp"
#include "memory/container/span.hpp"
#include <flatbuffers/detached_buffer.h>
namespace denox::compiler {

[[nodiscard]] flatbuffers::DetachedBuffer entry(memory::span<const std::byte> raw,
                                  const Options &options);

}

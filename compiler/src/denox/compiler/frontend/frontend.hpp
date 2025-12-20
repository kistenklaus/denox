#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/memory/container/span.hpp"

namespace denox::compiler {

Model frontend(memory::span<const std::byte> raw, const Options &options);

}



#pragma once

#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/memory/container/span.hpp"
namespace denox::compiler {

void reweight_dnx(memory::span<std::byte> dnx,
                      SuperGraph &supergraph);

} // namespace denox::compiler

#pragma once

#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/memory/container/span.hpp"
namespace denox::compiler {

SuperGraph select_dnx_edges(memory::span<const std::byte> dnx,
                      SuperGraph &supergraph);

} // namespace denox::compiler

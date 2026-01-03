#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/symbolic/SymGraphEval.hpp"
#include "denox/symbolic/SymGraph.hpp"

namespace denox::compiler {

SymGraphEval assumed_symeval(const SymGraph &symGraph,
                             memory::span<const NamedValue> valueNames,
                             const CompileOptions &options);

} // namespace denox::compiler

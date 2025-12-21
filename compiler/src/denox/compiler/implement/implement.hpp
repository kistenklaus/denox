#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/Supergraph.hpp"

namespace denox::compiler {

SuperGraph implement(const ConstModel &model, const SymGraph &symGraph,
                    const Options &options);
}

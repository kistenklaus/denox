#pragma once

#include "Options.hpp"
#include "compiler/ir/ConstModel.hpp"
#include "compiler/ir/impl/ImplModel.hpp"
#include "compiler/ir/populate/ImplDb.hpp"
#include "heuristic/IHeuristic.hpp"

namespace denox::compiler {

ImplDb implement_all(const OpModel &model, const SymGraph &symGraph,
                   const Options &options);
}

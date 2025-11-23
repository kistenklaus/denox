#pragma once

#include "Options.hpp"
#include "compiler/ir/ConstModel.hpp"
#include "compiler/ir/impl/ImplModel.hpp"
#include "compiler/ir/populate/ImplDb.hpp"

namespace denox::compiler {

ImplModel implement(const OpModel &model, const SymGraph& symGraph, const Options& options);

ImplDb implement_all(const OpModel &model, const SymGraph &symGraph,
                   const Options &options);
}

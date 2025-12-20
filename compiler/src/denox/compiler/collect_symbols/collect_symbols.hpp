#pragma once

#include "Options.hpp"
#include "compiler/ir/SymTable.hpp"
#include "model/Model.hpp"
namespace denox::compiler {

SymTable collect_symbols(const Model &model, const Options &options);

}

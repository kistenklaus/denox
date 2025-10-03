#pragma once

#include "Options.hpp"
#include "compiler/ir/SymTable.hpp"
#include "model/Model.hpp"
namespace denox::compiler {

SymTable sym_table(const Model &model, const Options &options);
}

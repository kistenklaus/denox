#pragma once

#include "compiler/ir/SymTable.hpp"
#include "compiler/ir/comp/CompModel.hpp"
#include "denox/symbolic/SymIR.hpp"
namespace denox::compiler {

std::pair<SymIR, std::uint32_t> compile_sym_and_remap(CompModel &model, SymTable& symTable);

} // namespace denox::compiler

#pragma once

#include "compiler/ir/comp/CompModel.hpp"
#include "symbolic/SymIR.hpp"
namespace denox::compiler {

SymIR sym_compile(const CompModel &model);

} // namespace denox::compiler

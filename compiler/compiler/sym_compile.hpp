#pragma once

#include "compiler/ir/comp/CompModel.hpp"
#include "symbolic/SymIR.hpp"
namespace denox::compiler {

std::pair<SymIR, std::uint32_t> compile_sym_and_remap(CompModel &model);

} // namespace denox::compiler

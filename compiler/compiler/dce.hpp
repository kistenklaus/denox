#pragma once

#include "compiler/ir/AdjModel.hpp"
#include "compiler/ir/CanoModel.hpp"
#include "compiler/ir/ConstModel.hpp"
#include "compiler/ir/SpecModel.hpp"
namespace denox::compiler {

/// Dead Code Elimination
OpModel dce(const SpecModel &model);

} // namespace denox::compiler

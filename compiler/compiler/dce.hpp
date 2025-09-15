#pragma once

#include "compiler/ir/AdjModel.hpp"
#include "compiler/ir/LinkedModel.hpp"
namespace denox::compiler {

/// Dead Code Elimination
AdjModel dce(const LinkedModel &model);

} // namespace denox::compiler

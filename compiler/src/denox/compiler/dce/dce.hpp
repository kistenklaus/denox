#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/specialization/SpecModel.hpp"

namespace denox::compiler {

/// Dead Code Elimination
ConstModel dce(const SpecModel &model);

} // namespace denox::compiler

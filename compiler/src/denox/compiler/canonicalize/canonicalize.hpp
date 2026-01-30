#pragma once

#include "denox/compiler/canonicalize/CanoModel.hpp"
#include "denox/common/ComputeOp.hpp"
#include "denox/compiler/frontend/model/Model.hpp"

namespace denox::compiler {

CanoModel canonicalize(const Model &model);

} // namespace denox::compiler

#pragma once

#include "compiler/ir/ConstModel.hpp"

namespace denox::compiler {

void implement(const OpModel &model, const SymGraph& symGraph);
}

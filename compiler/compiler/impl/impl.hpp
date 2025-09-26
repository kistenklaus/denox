#pragma once

#include "compiler/ir/ConstModel.hpp"
#include "compiler/ir/impl/ImplModel.hpp"

namespace denox::compiler {

ImplModel implement(const OpModel &model, const SymGraph& symGraph);
}

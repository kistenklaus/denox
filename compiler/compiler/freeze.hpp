#pragma once

#include "compiler/ir/AdjModel.hpp"
#include "compiler/ir/ConstModel.hpp"
namespace denox::compiler {

ConstModel freeze(const AdjModel &adjModel);
}

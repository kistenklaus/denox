#pragma once

#include "compiler/ir/comp/CompModel.hpp"
#include "compiler/ir/impl/ImplModel.hpp"

namespace denox::compiler {

CompModel placement(const ImplModel &model);

}

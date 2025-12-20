#pragma once

#include "Options.hpp"
#include "compiler/ir/comp/CompModel.hpp"
namespace denox::compiler {

void rebind_descriptors(CompModel &compModel, const Options& options);

}

#pragma once

#include "compiler/ir/comp/CompModel.hpp"
#include <dnx.h>

namespace denox::dnx {

flatbuffers::DetachedBuffer serialize(const compiler::CompModel &compModel,
                                      const compiler::SymIR &symIR);
}

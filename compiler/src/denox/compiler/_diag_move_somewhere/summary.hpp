#pragma once

#include "compiler/ir/comp/CompModel.hpp"
#include "compiler/ir/impl/ImplModel.hpp"
#include "flatbuffers/detached_buffer.h"
#include "model/Model.hpp"
#include "denox/symbolic/SymIR.hpp"
#include <fmt/printf.h>

namespace denox::compiler::diag {

void print_summary(const Model &model, const ImplModel &implModel,
                   const CompModel &compModel, const SymIR &symIR,
                   std::size_t symCount,
                   const flatbuffers::DetachedBuffer &dnx);

} // namespace denox::compiler

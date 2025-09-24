#pragma once

#include "compiler/ir/LinkedModel.hpp"
#include "memory/tensor/ActivationLayout.hpp"
namespace denox::compiler {

void specialize(LinkedModel &model, memory::span<const memory::ActivationLayout> layouts);

}

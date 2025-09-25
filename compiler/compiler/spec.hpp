#pragma once

#include "compiler/ir/CanoModel.hpp"
#include "compiler/ir/SpecModel.hpp"
#include "memory/tensor/ActivationLayout.hpp"
namespace denox::compiler {

SpecModel specialize(CanoModel &model,
                     memory::span<const memory::ActivationLayout> layouts);

}

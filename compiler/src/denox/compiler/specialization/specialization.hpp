#pragma once

#include "compiler/ir/CanoModel.hpp"
#include "compiler/ir/SpecModel.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"
namespace denox::compiler {

SpecModel specialize(CanoModel &model, const Lifetimes &lifetimes,
                     memory::span<const memory::ActivationLayout> layouts);

}

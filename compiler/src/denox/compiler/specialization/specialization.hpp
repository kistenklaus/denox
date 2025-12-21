#pragma once

#include "denox/compiler/canonicalize/CanoModel.hpp"
#include "denox/compiler/specialization/SpecModel.hpp"
#include "denox/memory/tensor/ActivationLayout.hpp"

namespace denox::compiler {

SpecModel specialize(CanoModel &model, const Lifetimes &lifetimes,
                     memory::span<const memory::ActivationLayout> layouts);

}

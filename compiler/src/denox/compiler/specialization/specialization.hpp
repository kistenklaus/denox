#pragma once

#include "denox/compiler/canonicalize/CanoModel.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/compiler/specialization/SpecModel.hpp"

namespace denox::compiler {

SpecModel specialize(const CanoModel &model, const Lifetimes &lifetimes);

}

#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/compile_shaders/SpvSchedule.hpp"
#include "denox/compiler/compile_symbols/SymProgram.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/compiler/frontend/model/NamedValue.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/symbolic/SymIR.hpp"

namespace denox::compiler {

SymProgram compile_symbols(SpvSchedule &schedule, const Model &model,
                           const Options &options);

} // namespace denox::compiler

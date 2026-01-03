#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/compile_shaders/SpvSchedule.hpp"
#include "denox/compiler/compile_symbols/SymProgram.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/memory/container/vector.hpp"

namespace denox::compiler {

memory::vector<std::byte> serialize(const compiler::SpvSchedule &schedule,
                                    const SymProgram &sprog,
                                    const Model& model,
                                    const CompileOptions &options);
}

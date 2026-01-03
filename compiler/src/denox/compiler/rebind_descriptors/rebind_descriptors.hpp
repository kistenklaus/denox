#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/compile_shaders/SpvSchedule.hpp"
#include "denox/spirv/SpirvTools.hpp"

namespace denox::compiler {

void rebind_descriptors(SpvSchedule &schedule, const CompileOptions &options, 
    spirv::SpirvTools* spirvTools);

}

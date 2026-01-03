#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/compile_shaders/SpvSchedule.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/compiler/placement/MemSchedule.hpp"
#include "denox/db/Db.hpp"
#include "denox/glsl/GlslCompiler.hpp"
namespace denox::compiler {

SpvSchedule compile_shaders(MemSchedule &&schedule, const Model &model, Db &db,
                            spirv::GlslCompiler *glslCompiler,
                            const CompileOptions &options);
}

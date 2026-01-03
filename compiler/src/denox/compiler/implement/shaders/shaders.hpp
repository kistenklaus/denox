#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/shaders/IShader.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include <memory>
#include <vector>

namespace denox::compiler::shaders {

std::vector<std::unique_ptr<IShader>>
get_all_shaders(spirv::GlslCompiler *compiler, const CompileOptions &options);
}

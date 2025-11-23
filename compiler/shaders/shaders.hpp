#pragma once

#include "Options.hpp"
#include "shaders/IShader.hpp"
#include <memory>
#include <vector>

namespace denox::compiler::shaders {

std::vector<std::unique_ptr<IShader>> get_all_shaders(GlslCompiler *compiler,
                                                      const Options &options);
}

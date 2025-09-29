#pragma once

#include "memory/container/string.hpp"

namespace denox::compiler::shaders {

memory::string preprocess_shader_src_pragmas(memory::string &src);

}

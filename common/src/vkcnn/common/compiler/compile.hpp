#pragma once

#include "vkcnn/common/compiler/CompileOptions.hpp"
#include "vkcnn/common/model/CompileModel.hpp"
#include "vkcnn/common/model/Model.hpp"
namespace vkcnn {

CompiledModel compile(Model model, const CompileOptions &options = {});

}

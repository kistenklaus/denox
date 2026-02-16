#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
namespace denox::compiler {

[[noreturn]] void failed_to_implement(const SuperGraph &supergraph, const ConstModel& model);
}

#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/Supergraph.hpp"

namespace denox::compiler {

void prune_dead_supergraph(SuperGraph &supergraph, const ConstModel& model);

}

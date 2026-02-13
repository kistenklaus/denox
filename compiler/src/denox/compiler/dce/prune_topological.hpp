#pragma once

#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/progress.hpp"
namespace denox::compiler {

void prune_topological(SuperGraph &supergraph, diag::Progress progess, diag::Logger& logger);

}

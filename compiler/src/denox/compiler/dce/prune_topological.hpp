#pragma once

#include "denox/compiler/dce/ConstModel.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/progress.hpp"
namespace denox::compiler {

void prune_topological(SuperGraph &supergraph, const ConstModel &model,
                       diag::Progress progess, const diag::Logger &logger);

}

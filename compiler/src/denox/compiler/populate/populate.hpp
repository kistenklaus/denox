#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/db/Db.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/progress.hpp"

namespace denox::compiler {

void populate(const compiler::SuperGraph &supergraph, Db &db,
              const SymGraphEval &symeval, diag::Progress progress, const diag::Logger &logger,
              const CompileOptions &options);

} // namespace denox::compiler

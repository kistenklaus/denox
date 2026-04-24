#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/frontend/model/Model.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
#include "denox/db/Db.hpp"
#include "denox/diag/progress.hpp"

namespace denox::compiler {

OptSchedule select_schedule(SuperGraph &&supergraph, const Db &db,
                            const Model &model, const SymGraphEval &symeval,
                            const CompileOptions &options,
                            diag::Progress progress,
                            const diag::Logger &logger);

} // namespace denox::compiler

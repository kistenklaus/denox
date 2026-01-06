#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/compiler/implement/Supergraph.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
#include "denox/db/Db.hpp"
#include "denox/compiler/frontend/model/Model.hpp"

namespace denox::compiler {

OptSchedule select_schedule(SuperGraph &&supergraph, const Db &db,
                            const Model &model, const CompileOptions &options,
                            diag::Logger& logger);

} // namespace denox::compiler

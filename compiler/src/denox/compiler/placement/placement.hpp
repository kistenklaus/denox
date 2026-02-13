#pragma once


#include "denox/compiler/placement/MemSchedule.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
#include "denox/diag/logging.hpp"
#include "denox/diag/progress.hpp"
namespace denox::compiler {

MemSchedule placement(const OptSchedule &schedule, diag::Progress progress, diag::Logger& logger);

}

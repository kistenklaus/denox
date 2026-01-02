#pragma once


#include "denox/compiler/placement/MemSchedule.hpp"
#include "denox/compiler/selection/OptSchedule.hpp"
namespace denox::compiler {

MemSchedule placement(const OptSchedule &schedule);
}

#pragma once

#include "pyvk/host/DispatchOp.hpp"
namespace pyvk {

void planMemory(std::span<DispatchOp> ops);
}

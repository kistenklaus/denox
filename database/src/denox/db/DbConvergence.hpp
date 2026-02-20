#pragma once

#include <cstdint>
namespace denox {

struct DbConvergenceInfo {
  uint64_t total_dispatch_count;
  uint64_t total_samples;
  uint64_t converged_min_dispatches;
  uint64_t converged_rel_dispatches;
};

} // namespace denox

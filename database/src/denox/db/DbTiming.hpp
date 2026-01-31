#pragma once

#include <cstdint>
#include <vector>
namespace denox {

struct DbSample {
  uint64_t timestamp;
  uint64_t latency_ns;
  uint32_t env;
};

struct DbDispatchTiming {
  std::vector<DbSample> samples;
  uint64_t mean_latency_ns;
  uint64_t std_derivation_ns;
};

} // namespace denox

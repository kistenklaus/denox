#pragma once

#include <cstdint>
namespace denox {

struct DbDispatchTiming {
  uint64_t samples;
  uint64_t latency_ns;
  uint64_t std_derivation_ns;
};

}


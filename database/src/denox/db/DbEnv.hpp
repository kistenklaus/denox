#pragma once

#include <cstdint>
#include <string>
namespace denox {

enum class DbClockMode {
  Unavailable,
  None,
  Base,
  Maximum,
};

struct DbEnv {
  std::string device;
  std::string os;
  std::string driver_version;
  std::string denox_version;
  std::string denox_commit_hash;

  uint64_t start_timestamp;

  DbClockMode clock_mode;
  uint16_t l2_warmup_iterations;
  uint16_t jit_warmup_iterations;
  uint16_t measurement_iterations;
};

} // namespace denox

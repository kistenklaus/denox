#pragma once

#include "denox/memory/container/vector.hpp"
#include <cstdint>

namespace denox::compiler {

struct Lifetime {
  std::uint64_t start;
  std::uint64_t end;
};

struct Lifetimes {
  memory::vector<Lifetime> valueLifetimes;
};

} // namespace denox::compiler

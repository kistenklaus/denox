#pragma once

#include <cstddef>
namespace vkcnn {

struct WeightTensorExtent {
  std::size_t k;
  std::size_t c;
  std::size_t r;
  std::size_t s;
};

} // namespace vkcnn

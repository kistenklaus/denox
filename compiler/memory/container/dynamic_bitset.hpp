#pragma once

#include <vector>
namespace denox::memory {

/// Requirements:
/// - Initalized to false.
/// - Doesn't have to be resizable or growing like a vector.
using dynamic_bitset = std::vector<bool>;

}

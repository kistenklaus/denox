#pragma once

namespace denox::memory {

using f64 = double;
using f64_reference = double &;
using f64_const_reference = const double &;
static_assert(sizeof(f64) == 8);

} // namespace denox::compiler

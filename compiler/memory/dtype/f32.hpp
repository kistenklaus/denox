#pragma once

namespace denox::memory {

using f32 = float;
using f32_reference = float &;
using f32_const_reference = const float &;
static_assert(sizeof(f32) == 4);

} // namespace denox::compiler

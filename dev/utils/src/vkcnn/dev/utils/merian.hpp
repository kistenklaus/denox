#pragma once

#include "merian/vk/context.hpp"

namespace vkcnn::merian {

/// Just a helper function, which does the initalization
::merian::ContextHandle createContext(std::string_view appName = "vkcnn");

} // namespace vkcnn::merian

#include "denox/common/version.hpp"
#include <fmt/format.h>

std::string denox::version() {
  return fmt::format("v{}.{}.{}", DENOX_VERSION_MAJOR, DENOX_VERSION_MINOR,
                     DENOX_VERSION_PATCH);
}

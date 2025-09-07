#pragma once

namespace denox::memory {

struct uvec2 {
  unsigned int x;
  unsigned int y;

  friend bool operator==(const uvec2& lhs, const uvec2& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
  }
  friend bool operator!=(const uvec2& lhs, const uvec2& rhs) {
    return !(lhs == rhs);
  }
};

} // namespace denox::memory

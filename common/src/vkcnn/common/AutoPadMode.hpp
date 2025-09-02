#pragma once

namespace vkcnn {

enum class AutoPadMode {
  None,
  SameUpper, // if input extent is odd, the extra pixel goes at the end (right/bottom)
  SameLower, // if input extent is odd, the extra pixel goes at the start (left/top)
  Zero,
};

}

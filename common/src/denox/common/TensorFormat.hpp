#pragma once

namespace denox {

enum class TensorFormat {
  Optimal,
  SSBO_HWC,
  SSBO_CHW,
  SSBO_CHWC8,
  TEX_RGBA,
  TEX_RGB,
  TEX_RG,
  TEX_R,
};

}

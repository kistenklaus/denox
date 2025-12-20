#pragma once

namespace denox {

enum class TensorStorage {
  Optimal,
  StorageBuffer,
  StorageImage,
  Sampler,
  CombinedImageSampler,
  SampledImage,
};

};

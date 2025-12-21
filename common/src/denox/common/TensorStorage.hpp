#pragma once
#include <fmt/core.h>

namespace denox {

enum class TensorStorage {
  Optimal,
  StorageBuffer,
  StorageImage,
  SampledStorageImage,
};

};

template <> struct fmt::formatter<denox::TensorStorage> {
  constexpr auto parse(fmt::format_parse_context &ctx) {
    return ctx.begin(); // no custom format specifiers
  }

  template <typename FormatContext>
  auto format(denox::TensorStorage storage, FormatContext &ctx) const {
    using enum denox::TensorStorage;

    std::string_view name;
    switch (storage) {
    case Optimal:
      name = "Optimal";
      break;
    case StorageBuffer:
      name = "StorageBuffer";
      break;
    case StorageImage:
      name = "StorageImage";
      break;
    case SampledStorageImage:
      name = "Sampler";
      break;
    default:
      name = "Unknown";
      break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};

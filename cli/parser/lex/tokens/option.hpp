#pragma once

#include <fmt/core.h>
#include <optional>
#include <string_view>

enum class OptionToken {
  SpirvOptimize,
  SpirvNonSemanticDebugInfo,
  SpirvDebugInfo,

  FeatureCoopmat,
  FeatureFusion,
  FeatureMemoryConcat,

  InputLayout,
  InputType,
  InputStorage,
  InputShape,

  OutputLayout,
  OutputType,
  OutputStorage,
  OutputShape,

  InputSize,

  TargetEnv,
  Device,

  Assume,

  Verbose,
  Quiet,

  Output,
  Input,

  Database,
};

std::optional<OptionToken> parse_option(std::string_view str);

template <> struct fmt::formatter<OptionToken> : fmt::formatter<std::string_view> {
  template <typename FormatContext>
  auto format(OptionToken opt, FormatContext &ctx) const {
    std::string_view name;

    switch (opt) {
    case OptionToken::SpirvOptimize:
      name = "spirv-optimize";
      break;
    case OptionToken::SpirvNonSemanticDebugInfo:
      name = "spirv-non-semantic-debug-info";
      break;
    case OptionToken::SpirvDebugInfo:
      name = "spirv-debug-info";
      break;

    case OptionToken::FeatureCoopmat:
      name = "feature-coopmat";
      break;
    case OptionToken::FeatureFusion:
      name = "feature-fusion";
      break;
    case OptionToken::FeatureMemoryConcat:
      name = "feature-memory-concat";
      break;

    case OptionToken::InputLayout:
      name = "input-layout";
      break;
    case OptionToken::InputType:
      name = "input-type";
      break;
    case OptionToken::InputStorage:
      name = "input-storage";
      break;
    case OptionToken::InputShape:
      name = "input-shape";
      break;

    case OptionToken::OutputLayout:
      name = "output-layout";
      break;
    case OptionToken::OutputType:
      name = "output-type";
      break;
    case OptionToken::OutputStorage:
      name = "output-storage";
      break;
    case OptionToken::OutputShape:
      name = "output-shape";
      break;

    case OptionToken::TargetEnv:
      name = "target-env";
      break;
    case OptionToken::Device:
      name = "device";
      break;

    case OptionToken::Assume:
      name = "assume";
      break;

    case OptionToken::Verbose:
      name = "verbose";
      break;
    case OptionToken::Quiet:
      name = "quiet";
      break;
    case OptionToken::Output:
      name = "output";
      break;
    case OptionToken::Input:
      name = "input";
      break;
    case OptionToken::Database:
      name = "database";
      break;
    case OptionToken::InputSize:
      name = "input-size";
      break;
    }

    return fmt::formatter<std::string_view>::format(name, ctx);
  }
};

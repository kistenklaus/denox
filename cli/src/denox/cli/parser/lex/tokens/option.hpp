#pragma once

#include <fmt/core.h>
#include <optional>
#include <string_view>

enum class OptionToken {
  SpirvOptimize,             //
  SpirvNonSemanticDebugInfo, //
  SpirvDebugInfo,            //
  FeatureCoopmat,            //
  FeatureFusion,             //
  FeatureMemoryConcat,       //

  Format,
  Storage,
  Shape,
  Type,

  InputSize,
  UseDescriptorSets,
  Specialize,
  MinSamples,
  MaxSamples,
  RelativeError,
  BatchSize,

  TargetEnv, //
  Device,    //
  Assume,    //
  Verbose,   //
  Quiet,     //
  Output,    //
  Input,
  Database, //

  OptimizationLevel,
  Jobs,
};

std::optional<OptionToken> parse_option(std::string_view str);

template <>
struct fmt::formatter<OptionToken> {

  constexpr auto parse(fmt::format_parse_context& ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(OptionToken opt, FormatContext& ctx) const {
    std::string_view name;

    switch (opt) {
      case OptionToken::SpirvOptimize: name = "spirv-optimize"; break;
      case OptionToken::SpirvNonSemanticDebugInfo: name = "spirv-non-semantic-debug-info"; break;
      case OptionToken::SpirvDebugInfo: name = "spirv-debug-info"; break;
      case OptionToken::FeatureCoopmat: name = "feature-coopmat"; break;
      case OptionToken::FeatureFusion: name = "feature-fusion"; break;
      case OptionToken::FeatureMemoryConcat: name = "feature-memory-concat"; break;
      case OptionToken::Format: name = "format"; break;
      case OptionToken::Storage: name = "storage"; break;
      case OptionToken::Shape: name = "shape"; break;
      case OptionToken::Type: name = "type"; break;
      case OptionToken::TargetEnv: name = "target-env"; break;
      case OptionToken::Device: name = "device"; break;
      case OptionToken::Assume: name = "assume"; break;
      case OptionToken::Verbose: name = "verbose"; break;
      case OptionToken::Quiet: name = "quiet"; break;
      case OptionToken::Output: name = "output"; break;
      case OptionToken::Input: name = "input"; break;
      case OptionToken::Database: name = "database"; break;
      case OptionToken::InputSize: name = "input-size"; break;
      case OptionToken::UseDescriptorSets: name = "use-descriptor-sets"; break;
      case OptionToken::Specialize: name = "specialize"; break;
      case OptionToken::MinSamples: name = "min-samples"; break;
      case OptionToken::MaxSamples: name = "max-samples"; break;
      case OptionToken::RelativeError: name = "relative-error"; break;
      case OptionToken::BatchSize: name = "batch-size"; break;
      case OptionToken::OptimizationLevel: name = "optimization-level"; break;
      case OptionToken::Jobs: name = "jobs"; break;
    }

    return fmt::format_to(ctx.out(), "{}", name);
  }
};

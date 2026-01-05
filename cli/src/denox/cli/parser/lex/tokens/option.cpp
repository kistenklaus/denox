#include "denox/cli/parser/lex/tokens/option.hpp"

std::optional<OptionToken> parse_option(std::string_view str) {
  if (str.empty()) {
    return std::nullopt;
  }

  // SPIR-V options
  if (str == "spirv-optimize" || str == "spv-opt") {
    return OptionToken::SpirvOptimize;
  }
  if (str == "spirv-non-semantic-debug-info") {
    return OptionToken::SpirvNonSemanticDebugInfo;
  }
  if (str == "spirv-debug-info") {
    return OptionToken::SpirvDebugInfo;
  }

  // feature toggles
  if (str == "feature-coopmat" || str == "fcoopmat") {
    return OptionToken::FeatureCoopmat;
  }
  if (str == "feature-fusion" || str == "ffusion") {
    return OptionToken::FeatureFusion;
  }
  if (str == "feature-memory-concat" || str == "fmemory-concat" ||
      str == "fmemcat") {
    return OptionToken::FeatureMemoryConcat;
  }

  // input specification
  if (str == "storage") {
    return OptionToken::Storage;
  }
  if (str == "format") {
    return OptionToken::Format;
  }
  if (str == "type") {
    return OptionToken::Type;
  }
  if (str == "shape") {
    return OptionToken::Shape;
  }

  // target / device
  if (str == "target-env") {
    return OptionToken::TargetEnv;
  }
  if (str == "device") {
    return OptionToken::Device;
  }

  // assumptions
  if (str == "assume") {
    return OptionToken::Assume;
  }

  // logging
  if (str == "verbose" || str == "v") {
    return OptionToken::Verbose;
  }
  if (str == "quiet" || str == "q") {
    return OptionToken::Quiet;
  }

  if (str == "input" || str == "i") {
    return OptionToken::Input;
  }

  if (str == "output" || str == "o") {
    return OptionToken::Output;
  }

  if (str == "database" || str == "db") {
    return OptionToken::Database;
  }

  if (str == "input-size") {
    return OptionToken::InputSize;
  }

  return std::nullopt;
}

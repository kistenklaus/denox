#include "CLI/CLI.hpp"
#include "denox/compiler.hpp"
#include <CLI/CLI.hpp>
#include <fmt/base.h>
#include <regex>

static std::regex shape_pattern_regex{
    "^([A-Za-z]+[A-Za-z0-9_]*|[0-9]+):([A-Za-z]+[A-Za-z0-9_]*|[0-9]+):([A-Za-z]"
    "+[A-Za-z]*|[0-9]+)$"};

int main(int argc, char **argv) {

  CLI::App app;
  denox::CompileOptions options;

  options.dnxVersion = 0;

  options.device.deviceName = "*RTX*";
  options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_4;

  options.features.coopmat = denox::Enable;
  app.add_flag(
      "--fcoopmat,!--fno-coopmat",
      [&](int x) {
        if (x > 0) {
          options.features.coopmat = denox::FeatureState::Require;
        } else if (x < 0) {
          options.features.coopmat = denox::FeatureState::Disable;
        } else {
          options.features.coopmat = denox::FeatureState::Enable;
        }
      },
      "Enable cooperative matrix shaders");

  options.features.fusion = denox::Enable;
  app.add_flag(
      "--ffusion,!--fno-fusion",
      [&](int x) {
        if (x > 0) {
          options.features.fusion = denox::FeatureState::Require;
        } else if (x < 0) {
          options.features.fusion = denox::FeatureState::Disable;
        } else {
          options.features.fusion = denox::FeatureState::Enable;
        }
      },
      "Enable layer fusion");

  options.features.memory_concat = denox::Enable;
  app.add_flag(
      "--fmemory-concat,!--fno-memory-concat",
      [&](int x) {
        if (x > 0) {
          options.features.memory_concat = denox::FeatureState::Require;
        } else if (x < 0) {
          options.features.memory_concat = denox::FeatureState::Disable;
        } else {
          options.features.memory_concat = denox::FeatureState::Enable;
        }
      },
      "Enable memory concat");

  options.spirvOptions.debugInfo = false;
  app.add_flag("--spirv-debug-info,!--spirv-no-debug-info",
               options.spirvOptions.debugInfo, "Generate SPIR-V debug info");

  options.spirvOptions.nonSemanticDebugInfo = false;
  app.add_flag(
      "--spirv-non-semantic-debug-info,!--no-spirv-non-semantic-debug-info",
      options.spirvOptions.nonSemanticDebugInfo,
      "Generates non semantic SPIR-V debug info. Sometimes useful for "
      "profiling tools.");

  options.spirvOptions.optimize = false;
  app.add_flag("--spirv-optimize,!--spirv-no-optimize",
               options.spirvOptions.optimize, "Optimize SPIR-V code");

  std::string modelFile;
  app.add_option("model", modelFile, "ONNX model file")
      ->check(CLI::ExistingFile)
      ->required();

  options.inputDescription.storage = denox::Storage::StorageBuffer;
  options.inputDescription.layout = denox::Layout::HWC;
  options.inputDescription.dtype = denox::DataType::Float16;
  options.outputDescription.storage = denox::Storage::StorageBuffer;
  options.outputDescription.layout = denox::Layout::HWC;
  options.outputDescription.dtype = denox::DataType::Float16;

  CLI::Validator shape_pattern{
      [&](const std::string &shape) {
        if (!std::regex_match(shape, shape_pattern_regex)) {
          return "invalid shape pattern";
        }
        return "";
      },
      "shape-pattern",
  };
  std::string input_shape;
  app.add_option("--input-shape", input_shape)->check(shape_pattern);

  CLI11_PARSE(app, argc, argv);

  { // input shape.
    std::sregex_iterator it(input_shape.begin(), input_shape.end(),
                            shape_pattern_regex);
    std::sregex_iterator end;
    while (it != end) {
      std::smatch match = *it;
      std::string match_str = match.str();
      // fmt::println("MATCH-STR: {}", match_str);
      ++it;
    }
  }

  denox::compile(modelFile.data(), options);

  return 0;
}

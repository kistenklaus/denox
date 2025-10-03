#include "CLI/CLI.hpp"
#include "denox/compiler.hpp"
#include <CLI/CLI.hpp>
#include <cctype>
#include <cstring>
#include <fmt/format.h>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

static std::regex shape_pattern_regex{
    "^([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    ":([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    ":([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    "$"};

static std::regex extent_pattern_regex{"^([A-Za-z0-9_]*)\\s*\\=?\\s*([0-9]*)$"};
static std::regex extent_infer_pattern_regex{"^\\s*\\?\\s*$"};

static std::vector<void *> g_deleteMe;

bool is_number(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

static denox::Extent parseExtent(const std::string &extentDef) {
  denox::Extent extent;
  extent.name = nullptr;
  extent.value = 0;

  std::smatch infer_match;
  if (std::regex_match(extentDef, infer_match, extent_infer_pattern_regex)) {
    return extent;
  }

  std::smatch match;
  if (std::regex_search(extentDef, match, extent_pattern_regex)) {
    std::string g1 = match[1].str();
    if (is_number(g1)) {
      int v = std::stoi(g1);
      extent.value = static_cast<unsigned int>(std::abs(v));
    } else if (match.length() >= 3 && is_number(match[2].str())) {
      std::string g2 = match[2].str();
      int v = std::stoi(g2);
      extent.value = static_cast<unsigned int>(std::abs(v));
      void *name = calloc(sizeof(std::string::value_type), g1.size() + 1);
      std::memcpy(name, g1.data(), g1.size());
      g_deleteMe.push_back(name);
      extent.name = static_cast<const char *>(name);
    } else {
      void *name = calloc(sizeof(std::string::value_type), g1.size() + 1);
      std::memcpy(name, g1.data(), g1.size());
      g_deleteMe.push_back(name);
      extent.name = static_cast<const char *>(name);
    }
  }
  return extent;
}

int main(int argc, char **argv) {

  CLI::App app;
  denox::CompileOptions options;

  options.dnxVersion = 0;

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

  std::optional<std::string> deviceName = std::nullopt;
  app.add_option("--device", deviceName, "Device name");

  CLI::Validator targetEnv_validator{
      [&](const std::string &env) {
        if (env == "vulkan1.0") {
          return "";
        } else if (env == "vulkan1.1") {
          return "";
        } else if (env == "vulkan1.2") {
          return "";
        } else if (env == "vulkan1.3") {
          return "";
        } else if (env == "vulkan1.4") {
          return "";
        } else {
          return "invalid target env";
        }
      },
      "target env pattern",
  };
  std::string targetEnv = "vulkan1.0";
  app.add_option("--target-env", targetEnv, "Target environment")
      ->check(targetEnv_validator);
  options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_0;

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
  std::string output_shape;
  app.add_option("--output-shape", output_shape)->check(shape_pattern);

  CLI11_PARSE(app, argc, argv);

  { // input shape.
    std::smatch match;
    if (std::regex_search(input_shape, match, shape_pattern_regex)) {
      std::string hstr = match[1].str();
      options.inputDescription.shape.height = parseExtent(hstr);
      std::string wstr = match[2].str();
      options.inputDescription.shape.width = parseExtent(wstr);
      std::string cstr = match[3].str();
      options.inputDescription.shape.channels = parseExtent(cstr);
    } else {
      std::memset(&options.inputDescription.shape, 0, sizeof(denox::Shape));
    }
  }
  { // output shape
    std::smatch match;
    if (std::regex_search(output_shape, match, shape_pattern_regex)) {
      std::string hstr = match[1].str();
      options.outputDescription.shape.height = parseExtent(hstr);
      std::string wstr = match[2].str();
      options.outputDescription.shape.width = parseExtent(wstr);
      std::string cstr = match[3].str();
      options.outputDescription.shape.channels = parseExtent(cstr);
    } else {
      std::memset(&options.outputDescription.shape, 0, sizeof(denox::Shape));
    }
  }

  options.device.deviceName = nullptr;
  if (deviceName.has_value()) {
    options.device.deviceName = deviceName->c_str();
  }
  if (targetEnv == "vulkan1.0") {
    options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_0;
  } else if (targetEnv == "vulkan1.1") {
    options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_1;
  } else if (targetEnv == "vulkan1.2") {
    options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_2;
  } else if (targetEnv == "vulkan1.3") {
    options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_3;
  } else if (targetEnv == "vulkan1.4") {
    options.device.apiVersion = denox::VulkanApiVersion::Vulkan_1_4;
  } else {
    throw std::invalid_argument("target-env argument is invalid");
  }

  denox::compile(modelFile.data(), options);

  for (void *p : g_deleteMe) {
    free(p);
  }

  return 0;
}

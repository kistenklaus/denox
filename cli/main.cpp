#include "CLI/CLI.hpp"
#include "denox/compiler.hpp"
#include <CLI/CLI.hpp>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

static std::vector<void *> g_deleteMe;

// TARGET-ENV
static CLI::Validator targetEnv_validator{
    [](const std::string &env) {
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
    " (vulkan1.0 | vulkan1.1 | vulkan1.2 | vulkan1.3 | vulkan1.4)",
};

static denox::VulkanApiVersion parseTargetEnv(const std::string &targetEnv) {
  if (targetEnv == "vulkan1.0" || targetEnv == "1.0") {
    return denox::VulkanApiVersion::Vulkan_1_0;
  } else if (targetEnv == "vulkan1.1" || targetEnv == "1.1") {
    return denox::VulkanApiVersion::Vulkan_1_1;
  } else if (targetEnv == "vulkan1.2" || targetEnv == "1.2") {
    return denox::VulkanApiVersion::Vulkan_1_2;
  } else if (targetEnv == "vulkan1.3" || targetEnv == "1.3") {
    return denox::VulkanApiVersion::Vulkan_1_3;
  } else if (targetEnv == "vulkan1.4" || targetEnv == "1.4") {
    return denox::VulkanApiVersion::Vulkan_1_4;
  } else {
    throw std::invalid_argument(
        fmt::format("Failed to parse target-env \"{}\"", targetEnv));
  }
}

// SHAPES
static std::regex shape_pattern_regex{
    "^([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    ":([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    ":([A-Za-z][A-Za-z0-9_]*|[A-Za-z][A-Za-z0-9_]*\\s*\\=\\s*[0-9]+|[0-9]+|\\?)"
    "$"};

static std::regex extent_pattern_regex{"^([A-Za-z0-9_]*)\\s*\\=?\\s*([0-9]*)$"};
static std::regex extent_infer_pattern_regex{"^\\s*\\?\\s*$"};

static CLI::Validator shape_pattern{
    [](const std::string &shape) {
      if (!std::regex_match(shape, shape_pattern_regex)) {
        return "invalid shape pattern";
      }
      return "";
    },
    ""};

static bool is_number(const std::string &s) {
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

static denox::Shape parseShape(const std::string &shape) {
  denox::Extent extents[3];
  std::smatch match;
  if (std::regex_search(shape, match, shape_pattern_regex)) {
    std::string hstr = match[1].str();
    extents[0] = parseExtent(hstr);
    std::string wstr = match[2].str();
    extents[1] = parseExtent(wstr);
    std::string cstr = match[3].str();
    extents[2] = parseExtent(cstr);
  } else {
    std::memset(extents, 0, sizeof(denox::Extent) * 3);
  }
  return denox::Shape{extents[0], extents[1], extents[2]};
}

// DATA-Type:
static CLI::Validator dtype_validator{
    [](const std::string &dtype) -> std::string {
      if (dtype == "f16" || dtype == "float16") {
        return "";
      } else if (dtype == "f32" || dtype == "float32") {
        return "";
      } else if (dtype == "u8" || dtype == "uint8") {
        return "";
      } else if (dtype == "i8" || dtype == "int8") {
        return "";
      } else if (dtype == "auto" || dtype == "?") {
        return "";
      }
      return fmt::format("\"{}\" is not a invalid or unsupported datatype",
                         dtype);
    },
    ""};
static denox::DataType parseDataType(const std::string &datatype) {
  if (datatype == "f16" || datatype == "float16") {
    return denox::DataType::Float16;
  } else if (datatype == "f32" || datatype == "float32") {
    return denox::DataType::Float32;
  } else if (datatype == "u8" || datatype == "uint8") {
    return denox::DataType::Uint8;
  } else if (datatype == "i8" || datatype == "int8") {
    return denox::DataType::Int8;
  } else if (datatype == "auto" || datatype == "?") {
    return denox::DataType::Auto;
  } else {
    throw std::invalid_argument(
        fmt::format("Failed to parse datatype \"{}\"", datatype));
  }
}

// LAYOUTS
static CLI::Validator layout_validator{
    [](const std::string &layout) -> std::string {
      if (layout == "HWC") {
        return "";
      } else if (layout == "CHW") {
        return "";
      } else if (layout == "CHWC8") {
        return "";
      }
      return fmt::format("Layout \"{}\" is either invalid or not supported",
                         layout);
    },
    ""};

static denox::Layout parseLayout(const std::string &layout) {
  if (layout == "HWC") {
    return denox::Layout::HWC;
  } else if (layout == "CHW") {
    return denox::Layout::CHW;
  } else if (layout == "CHWC8") {
    return denox::Layout::CHWC8;
  } else {
    throw std::invalid_argument(
        fmt::format("Failed to parse layout \"{}\"", layout));
  }
}

// STORAGE
static CLI::Validator storage_validator{
    [](const std::string &storage) -> std::string {
      if (storage == "SSBO" || storage == "buffer" ||
          storage == "storage-buffer") {
        return "";
      }
      return fmt::format("Invalid storage type \"{}\"", storage);
    },
    ""};
denox::Storage parseStorage(const std::string &storage) {
  if (storage == "SSBO" || storage == "buffer" || storage == "storage-buffer") {
    return denox::Storage::StorageBuffer;
  } else {
    throw std::invalid_argument(
        fmt::format("Failed to parse storage type \"{}\"", storage));
  }
}

int main(int argc, char **argv) {

  CLI::App app;
  denox::CompileOptions options;

  bool showVersion = false;
  app.add_flag("--version", showVersion, "Print version information");

  std::optional<std::string> modelFile;
  app.add_option("model", modelFile, "Path to ONNX model file")
      ->check(CLI::ExistingFile)
      ->type_name("");

  bool verbose = false;
  app.add_flag("--verbose,-v", verbose, "Print verbose log messages");
  bool quite = false;
  app.add_flag("--quite,-q", quite,
               "Disable all logging to the console. Errors are still shown");
  bool summarize = false;
  app.add_flag("--summarize", summarize,
               "Prints a summary of the execution plan");
  std::optional<std::string> artifactPath;
  app.add_option("-o", artifactPath, "Path to the produced dnx artifact.")
      ->type_name("");

  options.dnxVersion = 0;

  std::string features = "Features";

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
         "Enable/Disable shaders which use cooperative matrix.\nIf no flag is "
         "specified we "
         "query the driver for cooperative matrix support.")
      ->group(features);

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
         "Enable/Disable all non trivial layer fusions.\n"
         "By default all fusions are allowed.\nDisabling fusion will "
         "drastically "
         "decreases\nperformance; only disable for debugging reasons.")
      ->group(features);

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
         "Enable/Disable in-memory concat.\nconcat operations can be performed "
         "implicitly within the memory layout,\n"
         "for example if two tensors with a CHW layout are placed after each "
         "other in memory.\n"
         "This feature generally improves performance, only disable for "
         "debugging "
         "reasons.")
      ->group(features);

  std::string spirvOptions = "SPIRV compilation flags";

  options.spirvOptions.debugInfo = false;
  app.add_flag(
         "--spirv-debug-info,!--spirv-no-debug-info",
         options.spirvOptions.debugInfo,
         "Emit \"debugInfo\" within the shader compiler frontends and within "
         "SPIR-V tools.\n"
         "Importantly this doesn't enable non-semantic debug info, which might "
         "be "
         "interessting\n."
         "for profiling. By default debug info is stripped from the shaders.")
      ->group(spirvOptions);

  options.spirvOptions.nonSemanticDebugInfo = false;
  app.add_flag(
         "--spirv-non-semantic-debug-info,!--no-spirv-non-semantic-debug-info",
         options.spirvOptions.nonSemanticDebugInfo,
         "Emit non semantic SPIR-V debug info within the shader compilers "
         "frontend.\n"
         "Sometimes this is useful for profiling tools like NVIDIA Nsight.\n"
         "If not specified no non-semantic debug info is generated.")
      ->group(spirvOptions);

  options.spirvOptions.optimize = false;
  app.add_flag("--spirv-optimize,!--spirv-no-optimize",
               options.spirvOptions.optimize,
               "Enables/Disables spirv optimizations powered by spirv-tools.\n"
               "Specifically we run the spirv-tools optimizer until the spirv "
               "binary converges.\n"
               "By default no optimization passes are performed.")
      ->group(spirvOptions);

  options.spirvOptions.skipCompilation = false;
  app.add_flag("--spirv-skip-compilation", options.spirvOptions.skipCompilation,
               "Skip all shader compilation.")
      ->group(spirvOptions);

  std::string deviceInfo = "Target properties";
  std::optional<std::string> deviceName = std::nullopt;
  app.add_option(
         "--device", deviceName,
         "Device name pattern, which is used to query device capabilities.\n"
         "The pattern may contain .gitignore like expressions.\n"
         "For example: *RTX* or *AMD*\n"
         "If not specified we select the first discrete device.")
      ->type_name("")
      ->group(deviceInfo);

  std::string targetEnv = "vulkan1.1";
  app.add_option("--target-env", targetEnv,
                 "We require currently require vulkan1.1 to query subgroup "
                 "properties,\n"
                 "therefor we default to the api version 1.1")
      ->check(targetEnv_validator)
      ->type_name("")
      ->group(deviceInfo);
  options.outputDescription.storage = denox::Storage::StorageBuffer;

  std::string inputOutputGroup = "Input & Output";

  std::string input_shape;
  app.add_option("--input-shape", input_shape,
                 "Defines names for dynamic or specializes constant input "
                 "dimensions.\nEach extent seperated by \":\" can be either,"
                 "a C-style name,\na unsigned integral constant or a "
                 "assignment expression.\n"
                 "If not specified all dynamic extents are unnamed and \n"
                 "constant extents are inferred from the model file.")
      ->check(shape_pattern)
      ->group(inputOutputGroup)
      ->type_name("H:W:C");

  std::string inputType = "auto";
  app.add_option(
         "--input-type", inputType,
         "Overwrites the input type, if not specified the input type is\n"
         "inferred from the model. Supported types are: [f16, f32, u8, i8].")
      ->check(dtype_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  std::string inputLayout = "HWC";
  app.add_option(
         "--input-layout", inputLayout,
         "Specifies the input memory layout. Supported layouts are: [HWC, \n"
         "CHW, CHWC8]. HWC is the default layout, if not specified.")
      ->check(layout_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  std::string inputStorage = "SSBO";
  app.add_option("--input-storage", inputStorage,
                 "Specifies the storage class of the input. Supported storage "
                 "classes are:\n"
                 "[SSBO]. In not specified SSBO is chosen.")
      ->check(storage_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  std::string output_shape;
  app.add_option("--output-shape", output_shape,
                 "Defines names for dynamic or specializes "
                 "constant output dimensions.\nSemantic rules are equivalent "
                 "to \"--input-shape\"")
      ->check(shape_pattern)
      ->group(inputOutputGroup)
      ->type_name("H:W:C");

  std::string outputType = "auto";
  app.add_option(
         "--output-type", outputType,
         "Overwrites the output type, if not specified the output type is\n"
         "inferred from the model. Supported types are: [f16, f32, u8, i8].")
      ->check(dtype_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  std::string outputLayout = "HWC";
  app.add_option(
         "--output-layout", outputLayout,
         "Specifies the output memory layout. Supported layouts are: [HWC, \n"
         "CHW, CHWC8]. HWC is the default layout, if not specified.")
      ->check(layout_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  std::string outputStorage = "SSBO";
  app.add_option("--output-storage", outputStorage,
                 "Specifies the storage class of the input. Supported storage "
                 "classes are:\n"
                 "[SSBO]. In not specified SSBO is chosen.")
      ->check(storage_validator)
      ->type_name("")
      ->group(inputOutputGroup);

  CLI11_PARSE(app, argc, argv);

  options.inputDescription.shape = parseShape(input_shape);
  options.outputDescription.shape = parseShape(output_shape);
  options.inputDescription.dtype = parseDataType(inputType);
  options.outputDescription.dtype = parseDataType(outputType);
  options.inputDescription.layout = parseLayout(inputLayout);
  options.outputDescription.layout = parseLayout(outputLayout);
  options.inputDescription.storage = parseStorage(inputStorage);
  options.outputDescription.storage = parseStorage(outputStorage);

  options.device.deviceName =
      deviceName.has_value() ? deviceName->c_str() : nullptr;
  options.device.apiVersion = parseTargetEnv(targetEnv);

  options.verbose = verbose;
  options.quite = quite;
  options.summarize = summarize;

  if (showVersion) {
    fmt::println("denox version: 0.0.0");
  }
  if (modelFile.has_value()) {
    denox::CompilationResult result;
    if (denox::compile(modelFile->data(), &options, &result) < 0) {
      fmt::println("\x1B[31m[Error:]\x1B[0m {}", result.message);
    } else if (!options.spirvOptions.skipCompilation) {
      std::filesystem::path opath;
      if (artifactPath.has_value()) {
        opath = std::filesystem::current_path() / artifactPath.value();
      } else {
        opath = std::filesystem::path(modelFile.value());
        opath.replace_extension("dnx");
      }
      auto fstream = std::fstream(
          opath.string(), std::ios::out | std::ios::binary | std::ios::trunc);
      if (!fstream.is_open()) {
        fmt::println("\x1B[31m[Error:]\x1B[0m Failed to write dnx to \"{}\"",
                     opath.string());
      }
      fstream.write(static_cast<const char *>(result.dnx), result.dnxSize);
      fstream.close();
    }
    denox::destroy_compilation_result(&result);
  }
  if (!showVersion && !modelFile.has_value()) {
    fmt::println("no input model provided");
  }

  for (void *p : g_deleteMe) {
    free(p);
  }

  return 0;
}

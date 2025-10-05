#include "Options.hpp"
#include "denox/compiler.hpp"
#include "device_info/ApiVersion.hpp"
#include "device_info/DeviceInfo.hpp"
#include "device_info/query/query_driver_device_info.hpp"
#include "diag/logging.hpp"
#include "diag/not_implemented.hpp"
#include "diag/unreachable.hpp"
#include "entry.hpp"
#include "io/fs/File.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/span.hpp"
#include "memory/container/vector.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include "shaders/compiler/ShaderDebugInfoLevel.hpp"
#include "shaders/compiler/global_glslang_runtime.hpp"
#include <cassert>
#include <cstring>
#include <exception>
#include <fmt/format.h>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

namespace denox {

static memory::ActivationLayout parse_layout(Layout layout) {
  switch (layout) {
  case Layout::HWC:
    return memory::ActivationLayout::HWC;
  case Layout::CHW:
    return memory::ActivationLayout::CHW;
  case Layout::CHWC8:
    return memory::ActivationLayout::CHWC8;
  }
  compiler::diag::unreachable();
}

static constexpr std::size_t STABLE_DNX_VERSION = 1;

static unsigned infer_dnxVersion(const CompileOptions &options) {
  if (options.dnxVersion == 0) {
    return STABLE_DNX_VERSION;
  } else {
    return options.dnxVersion;
  }
}

static io::Path infer_cwd(const CompileOptions &options) {
  io::Path cwd;
  if (options.cwd == nullptr) {
    cwd = io::Path::cwd();
  } else {
    std::size_t cwdLen = std::strlen(options.cwd);
    memory::string_view cwdView{options.cwd, cwdLen};
    cwd = io::Path{cwdView}.absolute();
  }
  assert(cwd.is_absolute());
  return cwd;
}

static compiler::SrcType infer_srctype(const std::optional<io::Path> &srcPath,
                                       const CompileOptions &options) {
  switch (options.srcType) {
  case SrcType::Auto:
    if (!srcPath.has_value()) {
      throw std::runtime_error("Failed to infer src type");
    }
    assert(srcPath.has_value());
    if (srcPath->extension() == ".onnx") {
      return compiler::SrcType::Onnx;
    } else {
      throw std::runtime_error("Failed to infer src type");
    }
  case SrcType::Onnx:
    return compiler::SrcType::Onnx;
  }
  denox::compiler::diag::unreachable();
}

static memory::optional<memory::Dtype> parse_dtype(const DataType type) {
  switch (type) {
  case DataType::Auto:
    return memory::nullopt;
  case DataType::Float16:
    return memory::Dtype::F16;
  case DataType::Float32:
    return memory::Dtype::F32;
  case DataType::Uint8:
  case DataType::Int8:
    compiler::diag::not_implemented();
  }
  denox::compiler::diag::unreachable();
}

static compiler::TensorShapeDesc parse_shape(const Shape &shape) {
  compiler::TensorShapeDesc extent;
  if (shape.channels.name != nullptr) {
    extent.channels.name.emplace(shape.channels.name);
  }
  if (shape.channels.value != 0) {
    extent.channels.value = shape.channels.value;
  }
  if (shape.height.name != nullptr) {
    fmt::println("name= {}", shape.height.name);
    extent.height.name.emplace(shape.height.name);
  }
  if (shape.height.value != 0) {
    extent.height.value = shape.height.value;
  }
  if (shape.width.name != nullptr) {
    extent.width.name.emplace(shape.width.name);
  }
  if (shape.width.value != 0) {
    extent.width.value = shape.width.value;
  }
  return extent;
}

static compiler::DeviceInfo query_driver(const CompileOptions &options) {
  compiler::ApiVersion apiVersion;
  switch (options.device.apiVersion) {
  case VulkanApiVersion::Vulkan_1_0:
    apiVersion = compiler::ApiVersion::VULKAN_1_0;
    break;
  case VulkanApiVersion::Vulkan_1_1:
    apiVersion = compiler::ApiVersion::VULKAN_1_1;
    break;
  case VulkanApiVersion::Vulkan_1_2:
    apiVersion = compiler::ApiVersion::VULKAN_1_2;
    break;
  case VulkanApiVersion::Vulkan_1_3:
    apiVersion = compiler::ApiVersion::VULKAN_1_3;
    break;
  case VulkanApiVersion::Vulkan_1_4:
    apiVersion = compiler::ApiVersion::VULKAN_1_4;
    break;
  default:
    compiler::diag::unreachable();
  }

  memory::optional<memory::string> deviceName;
  if (options.device.deviceName != nullptr) {
    deviceName.emplace(options.device.deviceName);
  }

#ifdef DENOX_EXTERNALLY_MANAGED_VULKAN_CONTEXT
  if (options.externally_managed_vulkan_context != nullptr) {
    if (options.externally_managed_vulkan_context->instance != nullptr &&
        options.externally_managed_vulkan_context->physicalDevice != nullptr) {
      return compiler::device_info::query_driver_device_info(
          options.externally_managed_vulkan_context->instance,
          options.externally_managed_vulkan_context->physicalDevice,
          apiVersion);
    } else if (options.externally_managed_vulkan_context->instance != nullptr &&
               options.externally_managed_vulkan_context->physicalDevice ==
                   nullptr) {
      return compiler::device_info::query_driver_device_info(
          options.externally_managed_vulkan_context->instance, deviceName,
          apiVersion);
    }
  }
#endif

  return compiler::device_info::query_driver_device_info(apiVersion,
                                                         deviceName);
}

void compile(const char *cpath, const CompileOptions &options) {

  if (options.externally_managed_glslang_runtime) {
    compiler::global_glslang_runtime::assume_externally_managed();
  }

  compiler::DeviceInfo deviceInfo = query_driver(options);

  if (options.features.coopmat == Require &&
      deviceInfo.coopmat.supported == false) {
    DENOX_ERROR("Feature coopmat required, but not supported for device \"{}\"",
                deviceInfo.name);
    std::terminate();
  }
  if (options.features.coopmat == Disable) {
    deviceInfo.coopmat.supported = false;
  }

  compiler::FusionRules fusionRules;
  if (options.features.fusion != Disable) {
    fusionRules.enableSliceSliceFusion = true;
    fusionRules.enableConvReluFusion = true;
  }

  if (options.features.memory_concat != Disable) {
    fusionRules.enableImplicitConcat = true;
  }

  std::size_t pathLen = std::strlen(cpath);
  memory::string_view pathView{cpath, pathLen};
  io::Path path{pathView};

  if (path.empty()) {
    throw std::runtime_error("empty path to source");
  }
  if (!path.exists()) {
    throw std::runtime_error(
        fmt::format("source-file: {} does not exist.", path.str()));
  }

  auto dnxVersion = infer_dnxVersion(options);
  auto srcType = infer_srctype(path, options);
  auto cwd = infer_cwd(options);
  memory::ActivationLayout inputLayout =
      parse_layout(options.inputDescription.layout);
  memory::ActivationLayout outputLayout =
      parse_layout(options.outputDescription.layout);
  memory::optional<memory::Dtype> inputType =
      parse_dtype(options.inputDescription.dtype);
  memory::optional<memory::Dtype> outputType =
      parse_dtype(options.outputDescription.dtype);

  compiler::TensorShapeDesc inputShape =
      parse_shape(options.inputDescription.shape);
  compiler::TensorShapeDesc outputShape =
      parse_shape(options.outputDescription.shape);

  compiler::ShaderDebugInfoLevel spirvDebugInfoLevel =
      compiler::ShaderDebugInfoLevel::Strip;
  if (options.spirvOptions.debugInfo &&
      options.spirvOptions.nonSemanticDebugInfo) {
    spirvDebugInfoLevel =
        compiler::ShaderDebugInfoLevel::ForceNonSemanticDebugInfo;
  } else if (options.spirvOptions.debugInfo) {
    spirvDebugInfoLevel = compiler::ShaderDebugInfoLevel::Enable;
  } else {
    spirvDebugInfoLevel = compiler::ShaderDebugInfoLevel::Strip;
  }

  compiler::Options opt{
      .dnxVersion = dnxVersion,
      .srcType = srcType,
      .inputLayout = inputLayout,
      .inputType = inputType,
      .inputShape = inputShape,
      .outputLayout = outputLayout,
      .outputType = outputType,
      .outputShape = outputShape,
      .deviceInfo = std::move(deviceInfo),
      .fusionRules = fusionRules,
      .shaderDebugInfo = spirvDebugInfoLevel,
      .optimizeSpirv = options.spirvOptions.optimize,
      .cwd = cwd,
      .srcPath = path,
      .verbose = options.verbose,
      .quite = options.quite,
  };

  auto file = io::File::open(path, io::File::OpenMode::Read);
  memory::vector<std::byte> buf{file.size()};
  file.read_exact(buf);

  auto dnx = compiler::entry(buf, opt);

  // Write to file.
  io::File dnxFile =
      io::File::open(io::Path::cwd() / "net.dnx",
                     io::File::OpenMode::Write | io::File::OpenMode::Truncate);
  ;
  memory::span<const std::byte> dnxView{
      reinterpret_cast<const std::byte *>(dnx.data()), dnx.size()};
  dnxFile.write_exact(dnxView);
}

} // namespace denox

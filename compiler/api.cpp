#include "Options.hpp"
#include "denox/compiler.hpp"
#include "diag/unreachable.hpp"
#include "entry.hpp"
#include "io/fs/File.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/span.hpp"
#include "memory/container/vector.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include <cassert>
#include <cstring>
#include <fmt/format.h>
#include <stdexcept>

namespace denox {

static memory::ActivationLayout infer_layout(StorageLayoutFlags storageLayout) {
  if (storageLayout & LAYOUT_HWC) {
    return memory::ActivationLayout::HWC;
  } else {
    compiler::diag::unreachable();
  }
}

static unsigned infer_version(const CompileOptions &options) {
  if (options.version == 0) {
    return 1;
  } else {
    return options.version;
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
  case SrcType::ONNX:
    return compiler::SrcType::Onnx;
  }
  denox::compiler::diag::unreachable();
}

static std::optional<memory::Dtype> infer_dtype(const CoopMatElemType type) {
  switch (type) {
  case CoopMatElemType::float16_t:
    return memory::Dtype::F16;
  }
  denox::compiler::diag::unreachable();
}

static compiler::DeviceInfo infer_device_info(const CompileOptions &options) {
  if (options.deviceInfo != nullptr) {
    memory::vector<compiler::CoopMatType> coopmattypes;
    if (options.deviceInfo->coopmatTypes != nullptr) {
      coopmattypes.reserve(options.deviceInfo->coopmatTypeCount);
      for (std::size_t n = 0; n < options.deviceInfo->coopmatTypeCount; ++n) {
        const auto &coopmat = options.deviceInfo->coopmatTypes[n];
        if (coopmat.scope != CoopMatScope::Subgroup) {
          // ignore, denox, will probably not support those scopes for now.
          continue;
        }
        auto aType = infer_dtype(coopmat.aType);
        auto bType = infer_dtype(coopmat.bType);
        auto accType = infer_dtype(coopmat.accType);
        if (!aType.has_value() || !bType.has_value() || !accType.has_value()) {
          continue;
        }
        coopmattypes.emplace_back(coopmat.m, coopmat.k, coopmat.n, *aType,
                                  *bType, *accType);
      }
      coopmattypes.shrink_to_fit();
    }
    return compiler::DeviceInfo{
        .spirvVersion = options.deviceInfo->spirvVersion,
        .coopmatTypes = std::move(coopmattypes),
    };
  } else {
    // TODO create vulkan context and infer device info.
    throw std::logic_error("device info inferance is implemented!");
  }
}

static compiler::Features infer_features(const compiler::DeviceInfo &deviceInfo,
                                         const Features &features) {
  bool fusion{};
  switch (features.fusion) {
  case Require:
  case Enable:
    fusion = true;
    break;
  case Disable:
    fusion = false;
    break;
  default:
    compiler::diag::unreachable();
  }

  bool memory_concat{};
  switch (features.memory_concat) {
  case Require:
  case Enable:
    memory_concat = true;
    break;
  case Disable:
    memory_concat = false;
    break;
  default:
    compiler::diag::unreachable();
  }

  bool coopmat{};
  switch (features.coopmat) {
  case Require:
    if (deviceInfo.coopmatTypes.empty()) {
      throw std::runtime_error("Feature coopmat is required, but device does "
                               "not support cooperative matricies");
    }
    coopmat = true;
    break;
  case Enable:
    if (deviceInfo.coopmatTypes.empty()) {
      coopmat = false;
    } else {
      coopmat = true;
    }
    break;
  case Disable:
    coopmat = false;
    break;
  }

  return compiler::Features{
      .fusion = fusion,
      .memory_concat = memory_concat,
      .coopmat = coopmat,
  };
}

void compile(const char *cpath, const CompileOptions &options) {

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

  auto version = infer_version(options);
  auto srcType = infer_srctype(path, options);
  auto deviceInfo = infer_device_info(options);
  auto features = infer_features(deviceInfo, options.features);
  auto cwd = infer_cwd(options);
  auto srcPath = path;
  memory::ActivationLayout inputLayout =
      infer_layout(options.inputStorageLayout);
  memory::ActivationLayout outputLayout =
      infer_layout(options.outputStorageLayout);

  compiler::Options opt{
      .version = std::move(version),
      .srcType = std::move(srcType),
      .deviceInfo = std::move(deviceInfo),
      .features = std::move(features),
      .inputLayout = std::move(inputLayout),
      .outputLayout = std::move(outputLayout),
      .cwd = std::move(cwd),
      .srcPath = std::move(srcPath),
  };

  auto file = io::File::open(path, io::File::OpenMode::Read);
  memory::vector<std::byte> buf{file.size()};
  file.read_exact(buf);

  compiler::entry(buf, opt);
}

void compile(void *data, std::size_t n, const CompileOptions &options) {
  auto version = infer_version(options);
  auto srcType = infer_srctype(std::nullopt, options);
  auto deviceInfo = infer_device_info(options);
  auto features = infer_features(deviceInfo, options.features);
  auto cwd = infer_cwd(options);
  std::optional<io::Path> srcPath = std::nullopt;
  memory::ActivationLayout inputLayout =
      infer_layout(options.inputStorageLayout);
  memory::ActivationLayout outputLayout =
      infer_layout(options.outputStorageLayout);

  compiler::Options opt{

      .version = std::move(version),
      .srcType = std::move(srcType),
      .deviceInfo = std::move(deviceInfo),
      .features = std::move(features),
      .inputLayout = std::move(inputLayout),
      .outputLayout = std::move(outputLayout),
      .cwd = std::move(cwd),
      .srcPath = std::move(srcPath),
  };

  compiler::entry(
      memory::span<const std::byte>{static_cast<const std::byte *>(data), n},
      opt);
}

} // namespace denox

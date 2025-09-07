#include "entry.hpp"
#include "diag/logging.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/onnx.hpp"
#include "model/Model.hpp"
#include <fmt/base.h>
#include <spdlog/spdlog.h>

namespace denox::compiler {

static Model frontend(memory::span<const std::byte> raw,
                      const Options &options) {
  switch (options.srcType) {
  case SrcType::Onnx: {
    DENOX_INFO("Importing ONNX model");
    io::Path onnx_dir;
    if (options.srcPath.has_value()) {
      onnx_dir = options.srcPath->absolute().parent();
    } else {
      onnx_dir = options.cwd;
    }
    DENOX_DEBUG("External data directory: \"{}\"", onnx_dir.str());
    return denox::onnx::read(raw, onnx_dir);
  }
  }
  denox::compiler::diag::unreachable();
}

void entry(memory::span<const std::byte> raw, const Options &options) {
  compiler::diag::initLogger();

  DENOX_DEBUG("Run frontend");
  // 1. Import model
  Model model = frontend(raw, options);
  DENOX_DEBUG("Successfully imported Model:");
  DENOX_DEBUG_RAW(model.to_string());

  // 2. Preoptimizer pass: Trivial optimizations like Conv -> Conv, or Slice ->
  // Slice. Just some basic fusion not done by the exporters.

  // 3. Specialize graph: Specialize types and Layouts, possibly by duplicating.

  // 4. Build supergraph, based on available shaders.

  // 5. Generate shader source.

  // 6. Build DNX.
}

} // namespace denox::compiler

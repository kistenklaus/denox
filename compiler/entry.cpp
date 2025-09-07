#include "entry.hpp"
#include "diag/unreachable.hpp"
#include "frontend/onnx/onnx.hpp"
#include "model/Model.hpp"
#include <fmt/base.h>

namespace denox::compiler {

static Model frontend(memory::span<const std::byte> raw,
                      const Options &options) {
  switch (options.srcType) {
  case SrcType::Onnx: {
    io::Path onnx_dir;
    if (options.srcPath.has_value()) {
      onnx_dir = options.srcPath->parent();
    } else {
      onnx_dir = options.cwd;
    }
    return denox::onnx::read(raw, onnx_dir);
  }
  }
  denox::compiler::diag::unreachable();
}

void entry(memory::span<const std::byte> raw, const Options &options) {
  Model model = frontend(raw, options);
}
} // namespace denox::compiler

#pragma once
#include "denox/compiler/implement/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/symbolic/Sym.hpp"
#include <fmt/format.h>

namespace denox::compiler {

struct ComputeDispatchInfo {
  memory::optional<memory::string> name;
  memory::optional<memory::string> debug_info;
  memory::optional<io::Path> srcPath;
  memory::optional<Sym> memoryReads;
  memory::optional<Sym> memoryWrites;
};

struct ComputeDispatch {
  spirv::GlslCompilerInstance glsl;
  memory::small_vector<PushConstant, 6> pushConstants;
  Sym workgroupCountX;
  Sym workgroupCountY;
  Sym workgroupCountZ;
  memory::small_vector<TensorBinding, 4> bindings;

  ComputeDispatchInfo info;
};

} // namespace denox::compiler

template <> struct fmt::formatter<denox::compiler::ComputeDispatchInfo> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeDispatchInfo &info,
              FormatContext &ctx) const {
    auto out = ctx.out();
    *out++ = '{';

    bool first = true;
    auto emit = [&](const char *name, const auto &value) {
      if (!first) {
        *out++ = ',';
        *out++ = ' ';
      }
      out = fmt::format_to(out, "{}={}", name, value);
      first = false;
    };

    if (info.name)
      emit("name", *info.name);
    if (info.srcPath)
      emit("srcPath", *info.srcPath);
    if (info.memoryReads)
      emit("memoryReads", *info.memoryReads);
    if (info.memoryWrites)
      emit("memoryWrites", *info.memoryWrites);

    *out++ = '}';
    return out;
  }
};

template <> struct fmt::formatter<denox::compiler::ComputeDispatch> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const denox::compiler::ComputeDispatch &d,
              FormatContext &ctx) const {
    return fmt::format_to(ctx.out(),
                          "{{glsl_src={}, pushConstants={}, wgX={}, wgY={}, "
                          "wgZ={}, bindings={}, info={}}}",
                          d.glsl.getSourcePath(), d.pushConstants.size(),
                          d.workgroupCountX, d.workgroupCountY,
                          d.workgroupCountZ, d.bindings, d.info);
  }
};

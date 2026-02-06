#pragma once
#include "denox/common/PushConstant.hpp"
#include "denox/compiler/implement/TensorBinding.hpp"
#include "denox/glsl/GlslCompilerInstance.hpp"
#include "denox/io/fs/Path.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/container/string.hpp"
#include "denox/symbolic/Sym.hpp"
#include <fmt/format.h>

namespace denox::compiler {

struct ComputeDispatchInfo {
  memory::optional<memory::string> operation;
  memory::optional<memory::string> name;
  memory::optional<memory::string> config;
  memory::optional<io::Path> srcPath;
  memory::optional<Sym> memoryReads;
  memory::optional<Sym> memoryWrites;

  memory::optional<Sym> flops;

  memory::optional<bool> coopmat;

  memory::optional<memory::small_vector<uint32_t, 2>> input_bindings;
  memory::optional<memory::small_vector<uint32_t, 2>> output_bindings;
};

struct ComputeDispatch {
  static constexpr size_t PC_SVO = 4;
  static constexpr size_t BINDING_SVO = 4;
  spirv::GlslCompilerInstance glsl;
  memory::small_vector<PushConstant, PC_SVO> pushConstants;
  Sym workgroupCountX;
  Sym workgroupCountY;
  Sym workgroupCountZ;
  memory::small_vector<TensorBinding, BINDING_SVO> bindings;

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

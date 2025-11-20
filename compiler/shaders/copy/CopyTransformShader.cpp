#include "shaders/copy/CopyTransformShader.hpp"
#include "diag/logging.hpp"
#include "diag/unreachable.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include <stdexcept>

namespace denox::compiler::shader {}
denox::compiler::shaders::CopyTransformShader::CopyTransformShader(
    GlslCompiler *compiler, const Options &options)
    : m_compiler(compiler),
      m_enableImplicitConcat(options.fusionRules.enableImplicitConcat) {

  const auto supportedTensor = [](const TensorInstance &tensor) {
    if (tensor.type != memory::Dtype::F16) {
      return false;
    }
    if (tensor.layout != memory::ActivationLayout::CHWC8 &&
        tensor.layout != memory::ActivationLayout::HWC) {
      return false;
    }
    return true;
  };
  {
    Pattern concatPattern;
    auto concat = concatPattern.matchEdge();
    concat->matchRank(2);
    auto in0 = concat->matchSrc(0);
    auto in1 = concat->matchSrc(1);
    auto out = concat->matchDst();
    in0->matchValue(supportedTensor);
    in1->matchValue(supportedTensor);
    concat->matchValue(
        [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Concat; });
    m_patternHandles.emplace_back(in0, in1, out);
    m_capabilities.patterns.emplace_back(std::move(concatPattern),
                                         std::move(in0), std::move(in1),
                                         std::move(out));
  }
}

denox::memory::optional<unsigned int>
denox::compiler::shaders::CopyTransformShader::acceptMatch(
    const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
    [[maybe_unused]] unsigned int patternEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  unsigned int pattern = patternEnc & PATTERN_MASK;
  [[maybe_unused]] unsigned int mode = patternEnc & CONCAT_MODE_MASK;
  assert(mode == 0);

  const auto &patternHandles = m_patternHandles[pattern];
  const memory::NodeId src0Id = match[patternHandles.src0];
  const memory::NodeId src1Id = match[patternHandles.src1];
  const memory::NodeId dstId = match[patternHandles.dst];

  const TensorInstance &src0 = graph.get(src0Id);
  const TensorInstance &src1 = graph.get(src1Id);
  const TensorInstance &dst = graph.get(dstId);

  memory::ActivationLayout src0Layout =
      memory::ActivationLayout::demote(src0.layout, src0.channels);
  memory::ActivationLayout src1Layout =
      memory::ActivationLayout::demote(src1.layout, src1.channels);
  memory::ActivationLayout dstLayout =
      memory::ActivationLayout::demote(dst.layout, dst.channels);

  if (!(src0Layout == src1Layout && src1Layout == dstLayout)) {
    return memory::nullopt;
  }

  if (!m_enableImplicitConcat) {
    return pattern | EXPLICIT_CONCAT_MODE;
  }

  if (!(src0.layout == memory::ActivationLayout::CHWC8 &&
        src1.layout == memory::ActivationLayout::CHWC8 &&
        dst.layout == memory::ActivationLayout::CHWC8)) {
    return pattern | EXPLICIT_CONCAT_MODE;
  }
  if (!(src0.type == src1.type && src1.type == dst.type)) {
    return pattern | EXPLICIT_CONCAT_MODE;
  }
  assert(dst.channels == src0.channels + src1.channels);

  std::size_t src0ConcatFanout = 0;
  for (const auto &e : src0.originalNode->outgoing()) {
    if (e.value().tag() == ComputeOpTag::Concat) {
      ++src0ConcatFanout;
    }
  }

  std::size_t src1ConcatFanout = 0;
  for (const auto &e : src1.originalNode->outgoing()) {
    if (e.value().tag() == ComputeOpTag::Concat) {
      ++src1ConcatFanout;
    }
  }

  if (src0ConcatFanout > 1 && src1ConcatFanout > 1) {
    return pattern | EXPLICIT_CONCAT_MODE;
  }
  if (src0ConcatFanout == 1 && src1ConcatFanout == 1) {
    return pattern | IMPLICIT_CONCAT_MODE;
  }

  if (ENABLE_UNSTABLE_FEATURE_IMPLICIT_CONCAT_LIFETIME_INFERANCE) {
    const TensorInstance *aliasedSrc = nullptr;
    if (src0ConcatFanout == 1 && src1ConcatFanout != 1) {
      aliasedSrc = &src1;
    } else if (src1ConcatFanout == 1 && src0ConcatFanout != 1) {
      aliasedSrc = &src0;
    } else {
      compiler::diag::unreachable();
    }

    memory::vector<memory::NodeId> otherConcatDsts;
    for (const auto &e : aliasedSrc->originalNode->outgoing()) {
      if (e.value().tag() != ComputeOpTag::Concat)
        continue;
      const memory::NodeId did = e.dst().id();
      if (did == dst.originalNode->id())
        continue;
      otherConcatDsts.push_back(did);
    }

    bool allDisjoint = true;
    for (std::size_t k = 0; k < otherConcatDsts.size() && allDisjoint; ++k) {
      const std::uint64_t otherId = *otherConcatDsts[k];
      bool found = false;
      Lifetime otherLife{};
      for (std::size_t i = 0; i < graph.nodeCount(); ++i) {
        const TensorInstance &ti = graph.get(memory::NodeId(i));
        if (ti.originalNode->id() == memory::NodeId(otherId)) {
          otherLife = ti.lifetime;
          found = true;
          break;
        }
      }
      if (!found) {
        allDisjoint = false;
        break;
      }
      const bool disjointLifetimes = (dst.lifetime.end <= otherLife.start) ||
                                     (otherLife.end <= dst.lifetime.start);

      if (!disjointLifetimes)
        allDisjoint = false;
    }

    if (allDisjoint) {
      DENOX_WARN("Using unstable Feature: Implements a concat operation "
                 "implicitly in memory based on lifetime analysis. This may "
                 "currently result in unimplementable memory constraints (See "
                 "Github Issue #13)");
      return pattern | IMPLICIT_CONCAT_MODE; // viable implicit; actual anchor
                                             // chosen later
    }
  }

  return pattern | SINGLE_COPY_CONCAT_MODE;
}

float denox::compiler::shaders::CopyTransformShader::speedup(
    unsigned int patternEnc) const {
  switch (patternEnc & CONCAT_MODE_MASK) {
  case IMPLICIT_CONCAT_MODE:
    return 0.0f;
  case SINGLE_COPY_CONCAT_MODE:
    return 0.5f;
  case EXPLICIT_CONCAT_MODE:
    return 1.0f; // <- horrible performance
  default:
    compiler::diag::unreachable();
  }
}

void denox::compiler::shaders::CopyTransformShader::implement(
    Impl &impl, const memory::ConstGraph<TensorInstance, ComputeOp> &opGraph,
    unsigned int patternEnc,
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match,
    SymGraph &symGraph) const {
  unsigned int pattern = patternEnc & PATTERN_MASK;
  unsigned int mode = patternEnc & CONCAT_MODE_MASK;
  const auto &patternHandles = m_patternHandles[pattern];

  memory::NodeId src0Id = match[patternHandles.src0];
  memory::NodeId src1Id = match[patternHandles.src1];
  memory::NodeId dstId = match[patternHandles.dst];

  const auto &src0 = opGraph.get(src0Id);
  const auto &src1 = opGraph.get(src1Id);
  const auto &dst = opGraph.get(dstId);

  switch (mode) {
  case IMPLICIT_CONCAT_MODE: {
    impl.createImplicitConcatConstrain(src0Id, src1Id, dstId);
    break;
  }
  case SINGLE_COPY_CONCAT_MODE: {
    throw std::runtime_error("not implemented yet.");
  }
  case EXPLICIT_CONCAT_MODE: {
    assert(src0.layout == src1.layout);
    {
      // COPY SRC0
      std::uint32_t invocC;
      std::uint32_t invocW;
      std::uint32_t invocH;
      std::uint32_t wgC;
      std::uint32_t wgW;
      std::uint32_t wgH;

      auto shader = m_compiler->read(m_srcPath);
      shader.define("IN_CH_OFFSET", 0);
      shader.define("IN_CH", src0.channels);
      shader.define("OUT_CH_OFFSET", 0);
      shader.define("OUT_CH", dst.channels);
      if (src0.layout == memory::ActivationLayout::HWC &&
          dst.layout == memory::ActivationLayout::HWC &&
          (src0.channels % 8 != 0 || dst.channels % 8 != 0)) {
        shader.define("istype", "uint16_t");
        shader.define("ISTYPE_SIZE", 2);
        shader.define("ostype", "uint16_t");
        shader.define("OSTYPE_SIZE", 2);
        shader.define("IN_LAYOUT_HWC");
        shader.define("OUT_LAYOUT_HWC");

        if (src0.channels >= 16) {
          invocC = 2;
          invocW = 2;
          invocH = 1;
          wgC = 8;
          wgW = 32;
          wgH = 1;
        } else {
          invocC = 1;
          invocW = 4;
          invocH = 1;
          wgC = src0.channels;
          wgW = 32;
          wgH = 1;
        }
      } else if (src0.layout == memory::ActivationLayout::HWC &&
                 dst.layout == memory::ActivationLayout::HWC &&
                 (src0.channels % 8 == 0 && dst.channels % 8 == 0)) {
        shader.define("istype", "uvec4");
        shader.define("ISTYPE_SIZE", 16);
        shader.define("ostype", "uvec4");
        shader.define("OSTYPE_SIZE", 16);
        shader.define("IN_LAYOUT_HWC8");
        shader.define("OUT_LAYOUT_HWC8");

        if (src0.channels >= 32) {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 4;
          wgW = 64;
          wgH = 1;
        } else if (src0.channels >= 16) {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 2;
          wgW = 128;
          wgH = 1;
        } else {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 1;
          wgW = 256;
          wgH = 1;
        }
      } else if (src0.layout == memory::ActivationLayout::CHWC8 &&
                 dst.layout == memory::ActivationLayout::CHWC8) {
        shader.define("istype", "uvec4");
        shader.define("ISTYPE_SIZE", 16);
        shader.define("ostype", "uvec4");
        shader.define("OSTYPE_SIZE", 16);
        shader.define("IN_LAYOUT_CHWC8");
        shader.define("OUT_LAYOUT_CHWC8");
        {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 1;
          wgW = 256;
          wgH = 1;
        }
      } else {
        compiler::diag::unreachable();
      }

      shader.define("INVOC_C", invocC);
      shader.define("INVOC_W", invocW);
      shader.define("INVOC_H", invocH);
      shader.define("WG_C", wgC);
      shader.define("WG_W", wgW);
      shader.define("WG_H", wgH);

      std::uint32_t tileX = invocC * wgC;
      std::uint32_t tileY = invocW * wgW;
      std::uint32_t tileZ = invocH * wgH;

      Sym workgroupCountX = symGraph.cdiv(src0.channels, tileX);
      Sym workgroupCountY = symGraph.cdiv(src0.extent.x.asSym(), tileY);
      Sym workgroupCountZ = symGraph.cdiv(src0.extent.y.asSym(), tileZ);

      auto copySrc0Dispatch = impl.registerDispatch(
          std::move(shader), workgroupCountX, workgroupCountY, workgroupCountZ);
      copySrc0Dispatch.addBinding(0, 0, AccessFlag::ReadOnly, src0Id);
      copySrc0Dispatch.addBinding(0, 1, AccessFlag::WriteOnly, dstId);
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.x));
      copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.y));
      copySrc0Dispatch.setName("explicit-concat-copy-src0");
      copySrc0Dispatch.setSourcePath(m_srcPath);

      Sym reads = symGraph.mul(src0.extent.x.asSym(), src0.extent.y.asSym(),
                               src0.channels * src0.type.size());
      Sym writes = symGraph.mul(src0.extent.x.asSym(), src0.extent.y.asSym(),
                                src0.channels * dst.type.size());
      copySrc0Dispatch.setMemoryReads(reads);
      copySrc0Dispatch.setMemoryWrites(writes);
      copySrc0Dispatch.setDebugInfo(fmt::format("CopyTransformShader\n"
                                                "- IN_LAYOUT:  {}\n"
                                                "- OUT_LAYOUT: {}\n",
                                                src0.layout.to_string(),
                                                dst.layout.to_string()));

      copySrc0Dispatch.setInputDesc(
          fmt::format("{}[{}]", src0.layout.to_string(), src0.channels));
      copySrc0Dispatch.setOutputDesc(
          fmt::format("{}[{}]", dst.layout.to_string(), dst.channels));
    }

    // COPY SRC1
    {
      auto shader = m_compiler->read(m_srcPath);
      std::uint32_t invocC;
      std::uint32_t invocW;
      std::uint32_t invocH;
      std::uint32_t wgC;
      std::uint32_t wgW;
      std::uint32_t wgH;

      shader.define("IN_CH_OFFSET", 0);
      shader.define("IN_CH", src1.channels);
      shader.define("OUT_CH_OFFSET", src0.channels);
      shader.define("OUT_CH", dst.channels);
      if (src0.layout == memory::ActivationLayout::HWC &&
          dst.layout == memory::ActivationLayout::HWC &&
          (src0.channels % 8 != 0 || src1.channels % 8 != 0 ||
           dst.channels % 8 != 0)) {
        shader.define("istype", "uint16_t");
        shader.define("ISTYPE_SIZE", 2);
        shader.define("ostype", "uint16_t");
        shader.define("OSTYPE_SIZE", 2);
        shader.define("IN_LAYOUT_HWC");
        shader.define("OUT_LAYOUT_HWC");

        if (src1.channels >= 16) {
          invocC = 2;
          invocW = 2;
          invocH = 1;
          wgC = 8;
          wgW = 32;
          wgH = 1;
        } else {
          invocC = 1;
          invocW = 4;
          invocH = 1;
          wgC = src1.channels;
          wgW = 32;
          wgH = 1;
        }
      } else if (src0.layout == memory::ActivationLayout::HWC &&
                 dst.layout == memory::ActivationLayout::HWC &&
                 (src0.channels % 8 == 0 && src1.channels % 8 == 0 &&
                  dst.channels % 8 == 0)) {
        shader.define("istype", "uvec4");
        shader.define("ISTYPE_SIZE", 16);
        shader.define("ostype", "uvec4");
        shader.define("OSTYPE_SIZE", 16);
        shader.define("IN_LAYOUT_HWC8");
        shader.define("OUT_LAYOUT_HWC8");

        if (src1.channels >= 32) {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 4;
          wgW = 64;
          wgH = 1;
        } else if (src1.channels >= 16) {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 2;
          wgW = 128;
          wgH = 1;
        } else {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 1;
          wgW = 256;
          wgH = 1;
        }
      } else if (src0.layout == memory::ActivationLayout::CHWC8 &&
                 dst.layout == memory::ActivationLayout::CHWC8) {
        shader.define("istype", "uvec4");
        shader.define("ISTYPE_SIZE", 16);
        shader.define("ostype", "uvec4");
        shader.define("OSTYPE_SIZE", 16);
        shader.define("IN_LAYOUT_CHWC8");
        shader.define("OUT_LAYOUT_CHWC8");
        {
          invocC = 8;
          invocW = 1;
          invocH = 1;
          wgC = 1;
          wgW = 256;
          wgH = 1;
        }
      } else {
        compiler::diag::unreachable();
      }
      shader.define("INVOC_C", invocC);
      shader.define("INVOC_W", invocW);
      shader.define("INVOC_H", invocH);
      shader.define("WG_C", wgC);
      shader.define("WG_W", wgW);
      shader.define("WG_H", wgH);

      std::uint32_t tileX = invocC * wgC;
      std::uint32_t tileY = invocW * wgW;
      std::uint32_t tileZ = invocH * wgH;

      Sym workgroupCountX = symGraph.cdiv(src1.channels, tileX);
      Sym workgroupCountY = symGraph.cdiv(src1.extent.x.asSym(), tileY);
      Sym workgroupCountZ = symGraph.cdiv(src1.extent.y.asSym(), tileZ);

      auto copySrc1Dispatch = impl.registerDispatch(
          std::move(shader), workgroupCountX, workgroupCountY, workgroupCountZ);
      copySrc1Dispatch.addBinding(0, 0, AccessFlag::ReadOnly, src1Id);
      copySrc1Dispatch.addBinding(0, 1, AccessFlag::WriteOnly, dstId);
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.x));
      copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.y));
      copySrc1Dispatch.setName("explicit-concat-copy-src1");
      copySrc1Dispatch.setSourcePath(m_srcPath);

      Sym reads = symGraph.mul(src1.extent.x.asSym(), src1.extent.y.asSym(),
                               src1.channels * src1.type.size());
      Sym writes = symGraph.mul(src1.extent.x.asSym(), src1.extent.y.asSym(),
                                src1.channels * dst.type.size());
      copySrc1Dispatch.setMemoryReads(reads);
      copySrc1Dispatch.setMemoryWrites(writes);
      copySrc1Dispatch.setDebugInfo(fmt::format("CopyTransformShader\n"
                                                "- IN_LAYOUT:  {}\n"
                                                "- OUT_LAYOUT: {}\n",
                                                src1.layout.to_string(),
                                                dst.layout.to_string()));

      copySrc1Dispatch.setInputDesc(
          fmt::format("{}[{}]", src1.layout.to_string(), src1.channels));
      copySrc1Dispatch.setOutputDesc(
          fmt::format("{}[{}]", dst.layout.to_string(), dst.channels));
    }
    break;
  }
  }
}

denox::memory::string denox::compiler::shaders::CopyTransformShader::name(
    unsigned int patternEnc) const {
  switch (patternEnc & CONCAT_MODE_MASK) {
  case IMPLICIT_CONCAT_MODE:
    return "implicit-concat";
  case SINGLE_COPY_CONCAT_MODE:
    return "single-copy-concat";
  case EXPLICIT_CONCAT_MODE:
    return "explicit-concat";
  default:
    compiler::diag::unreachable();
  }
}

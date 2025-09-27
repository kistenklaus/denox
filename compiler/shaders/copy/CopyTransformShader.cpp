#include "shaders/copy/CopyTransformShader.hpp"
#include "diag/logging.hpp"
#include "diag/unreachable.hpp"
#include <stdexcept>

namespace denox::compiler::shader {

}
denox::compiler::shaders::CopyTransformShader::CopyTransformShader(
    GlslCompiler *compiler)
    : m_compiler(compiler) {
  {
    Pattern concatPattern;
    auto concat = concatPattern.matchEdge();
    concat->matchRank(2);
    auto in0 = concat->matchSrc(0);
    auto in1 = concat->matchSrc(1);
    auto out = concat->matchDst();
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

  if (!(src0.layout == src1.layout && src1.layout == dst.layout)) {
    return memory::nullopt;
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
      const std::uint64_t otherId = otherConcatDsts[k];
      bool found = false;
      Lifetime otherLife{};
      for (std::size_t i = 0; i < graph.nodeCount(); ++i) {
        const TensorInstance &ti = graph.get(memory::NodeId(i));
        if (ti.originalNode->id() == otherId) {
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
    const algorithm::ConstGraphMatch<TensorInstance, ComputeOp> &match) const {
  unsigned int pattern = patternEnc & PATTERN_MASK;
  unsigned int mode = patternEnc & CONCAT_MODE_MASK;
  const auto &patternHandles = m_patternHandles[pattern];

  memory::NodeId src0Id = match[patternHandles.src0];
  memory::NodeId src1Id = match[patternHandles.src1];
  memory::NodeId dstId = match[patternHandles.dst];

  const auto &src0 = opGraph.get(src0Id);
  const auto &src1 = opGraph.get(src1Id);

  switch (mode) {
  case IMPLICIT_CONCAT_MODE: {
    impl.createImplicitConcatConstrain(src0Id, src1Id, dstId);
    break;
  }
  case SINGLE_COPY_CONCAT_MODE: {
    [[maybe_unused]] auto dispatch = impl.dispatch({});
    // TODO Select anchor.
    throw std::runtime_error("not implemented yet.");

    break;
  }
  case EXPLICIT_CONCAT_MODE: {
    auto copySrc0Dispatch = impl.dispatch({});
    copySrc0Dispatch.addBinding(src0Id);
    copySrc0Dispatch.addBinding(dstId);
    copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.x));
    copySrc0Dispatch.addPushConstant(PushConstant::Dynamic(src0.extent.y));
    copySrc0Dispatch.setName("explicit-concat-copy-src0");
    copySrc0Dispatch.setSourcePath(m_srcPath);

    auto copySrc1Dispatch = impl.dispatch({});
    copySrc1Dispatch.addBinding(src1Id);
    copySrc1Dispatch.addBinding(dstId);
    copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.x));
    copySrc1Dispatch.addPushConstant(PushConstant::Dynamic(src1.extent.y));
    copySrc1Dispatch.setName("explicit-concat-copy-src1");
    copySrc1Dispatch.setSourcePath(m_srcPath);
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


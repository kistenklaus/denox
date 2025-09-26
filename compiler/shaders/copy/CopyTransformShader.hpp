#pragma once

#include "algorithm/pattern_matching/GraphPattern.hpp"
#include "diag/unreachable.hpp"
#include "shaders/IShader.hpp"
namespace denox::compiler::shaders {

class CopyTransformShader : public IShader {
public:
  using Pattern = algorithm::GraphPattern<TensorInstance, ComputeOp>;

  static constexpr unsigned int EXPLICIT_CONCAT_PATTERN = 0;
  static constexpr unsigned int IMPLICIT_CONCAT_PATTERN = 1;
  static constexpr unsigned int SINGLE_COPY_CONCAT_PATTERN = 2;

  CopyTransformShader() {
    {
      Pattern concatPattern;
      auto concat = concatPattern.matchEdge();
      concat->matchRank(2);
      auto in0 = concat->matchSrc(0);
      auto in1 = concat->matchSrc(1);
      auto out = concat->matchDst();
      concat->matchValue(
          [](const ComputeOp &op) { return op.tag() == ComputeOpTag::Concat; });
      m_concatSrc0 = in0;
      m_concatSrc1 = in1;
      m_concatDst = out;

      m_capabilities.patterns.emplace_back(std::move(concatPattern),
                                           std::move(in0), std::move(in1),
                                           std::move(out));
    }
  }

  const ShaderCapabilities &capabilities() const final override {
    return m_capabilities;
  }

  memory::optional<unsigned int>
  acceptMatch(const memory::ConstGraph<TensorInstance, ComputeOp> &graph,
              [[maybe_unused]] unsigned int pattern,
              const algorithm::ConstGraphMatch<TensorInstance, ComputeOp>
                  &match) const final override {
    const memory::NodeId src0Id = match[m_concatSrc0];
    const memory::NodeId src1Id = match[m_concatSrc1];
    const memory::NodeId dstId = match[m_concatDst];

    const TensorInstance &src0 = graph.get(src0Id);
    const TensorInstance &src1 = graph.get(src1Id);
    const TensorInstance &dst = graph.get(dstId);

    if (!(src0.layout == src1.layout && src1.layout == dst.layout)) {
      return memory::nullopt;
    }

    if (!(src0.layout == memory::ActivationLayout::CHWC8 &&
          src1.layout == memory::ActivationLayout::CHWC8 &&
          dst.layout == memory::ActivationLayout::CHWC8)) {
      return EXPLICIT_CONCAT_PATTERN;
    }
    if (!(src0.type == src1.type && src1.type == dst.type)) {
      return EXPLICIT_CONCAT_PATTERN;
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
      return EXPLICIT_CONCAT_PATTERN;
    }
    if (src0ConcatFanout == 1 && src1ConcatFanout == 1) {
      return IMPLICIT_CONCAT_PATTERN;
    }

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
      return IMPLICIT_CONCAT_PATTERN; // viable implicit; actual anchor chosen
                                      // later
    }

    return SINGLE_COPY_CONCAT_PATTERN;
  }

  float speedup([[maybe_unused]] unsigned int pattern) const final override {
    switch (pattern) {
    case IMPLICIT_CONCAT_PATTERN:
      return 0.0f;
    case SINGLE_COPY_CONCAT_PATTERN:
      return 0.5f;
    case EXPLICIT_CONCAT_PATTERN:
      return 1.0f;
    default:
      compiler::diag::unreachable();
    }
  }

  // TODO Figure out the return from here, maybe directly somethig like a
  // dispatch with a compiled SPIR-V or something like this.
  void implement([[maybe_unused]] unsigned int pattern,
                 [[maybe_unused]] const algorithm::ConstGraphMatch<
                     TensorInstance, ComputeOp> &match) const final override {}

  memory::string name(unsigned int pattern) const final override {
    switch (pattern) {
    case IMPLICIT_CONCAT_PATTERN:
      return "implicit-concat";
    case SINGLE_COPY_CONCAT_PATTERN:
      return "single-copy-concat";
    case EXPLICIT_CONCAT_PATTERN:
      return "explicit-concat";
    default:
      compiler::diag::unreachable();
    }
  }

private:
  ShaderCapabilities m_capabilities;
  Pattern::NP m_concatSrc0;
  Pattern::NP m_concatSrc1;
  Pattern::NP m_concatDst;
};

} // namespace denox::compiler::shaders

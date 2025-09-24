#pragma once

#include "algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "compiler/cano/rules/IFusionRule.hpp"
#include "diag/logging.hpp"
#include "memory/container/optional.hpp"
#include "model/ComputeOp.hpp"
#include <cassert>
namespace denox::compiler::cano {

class SliceSlice : public IFusionRule {
private:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;
  struct StaticSliceSliceSingleton {
    Pattern pattern;
    Pattern::NP A;
    Pattern::EP AB;
    Pattern::NP B;
    Pattern::EP BC;
    Pattern::NP C;
  };

  static const StaticSliceSliceSingleton &getInstance() {
    static StaticSliceSliceSingleton singleton = []() {
      Pattern pattern;
      auto A = pattern.matchNode();
      auto AB = A->matchOutgoing();
      auto B = AB->matchDst();
      auto BC = B->matchOutgoing();
      auto C = BC->matchDst();

      AB->matchRank(1);
      BC->matchRank(1);

      B->matchOutDeg(1);
      B->matchInDeg(1);

      static auto isSliceOp = [](const ComputeOp &op) {
        return op.tag() == ComputeOpTag::Slice;
      };

      AB->matchValue(isSliceOp);
      BC->matchValue(isSliceOp);

      return StaticSliceSliceSingleton{
          //
          .pattern = std::move(pattern), .A = std::move(A),
          .AB = std::move(AB),           .B = std::move(B),
          .BC = std::move(BC),           .C = std::move(C)};
    }();
    return singleton;
  }

public:
  const algorithm::GraphPattern<ComputeTensor, ComputeOp> &
  pattern() final override {
    return getInstance().pattern;
  }

  virtual void apply(const algorithm::LinkedGraphMatch<ComputeTensor, ComputeOp>
                         &match) final override {
    const auto &self = getInstance();

    auto nodeA = match[self.A];
    auto nodeC = match[self.C];
    auto edgeAB = match[self.AB];
    auto edgeBC = match[self.BC];

    assert(edgeAB.value().tag() == ComputeOpTag::Slice);
    assert(edgeBC.value().tag() == ComputeOpTag::Slice);

    const auto &slice1 = edgeAB.value().slice();
    const auto &slice2 = edgeBC.value().slice();

    const auto resolveAxis = [](const Sym &lhs,
                                const Sym &rhs) -> memory::optional<Sym> {
      if (lhs.isSymbolic() && rhs.isSymbolic()) {
        return memory::nullopt;
      }
      if (lhs.isConstant() && rhs.isConstant()) {
        return Sym::Const(
            std::min<Sym::value_type>(lhs.constant(), rhs.isConstant()));
      }
      if (lhs.isConstant() && rhs.isSymbolic()) {
        if (lhs.constant() != 0) {
          return memory::nullopt;
        }
        return rhs;
      }
      if (lhs.isSymbolic() && rhs.isConstant()) {
        if (lhs.constant() != 0) {

          return memory::nullopt;
        }
        return lhs;
      }
      return memory::nullopt;
    };

    memory::optional<Sym> left = resolveAxis(slice1->left, slice2->left);
    memory::optional<Sym> right = resolveAxis(slice1->right, slice2->right);
    memory::optional<Sym> top = resolveAxis(slice1->top, slice2->top);
    memory::optional<Sym> bottom = resolveAxis(slice1->bottom, slice2->bottom);

    if (!left.has_value() || !right.has_value() || !top.has_value() ||
        !bottom.has_value()) {
      DENOX_WARN("Detected possible fusion of two slice operations duration "
                 "cano pass, but failed to implement it.");
      return; // bail!
    }

    ComputeOp slice = ComputeOpSlice(*left, *right, *top, *bottom);

    nodeA->outgoing().insert_after(edgeAB.nextOutgoingIterator(), nodeC,
                                   std::move(slice));

    edgeAB.erase();
  }

private:
};

} // namespace denox::compiler::cano

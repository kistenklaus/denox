#pragma once

#include "algorithm/pattern_matching/LinkedGraphMatch.hpp"
#include "compiler/cano/rules/IFusionRule.hpp"
#include "diag/logging.hpp"
#include "memory/container/optional.hpp"
#include "model/ComputeOp.hpp"
#include "symbolic/SymGraph.hpp"
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

    nodeA->outgoing().insert_after(edgeAB.nextOutgoingIterator(), nodeC,
                                   edgeBC.value().slice());
    edgeAB.erase();
  }

private:
};

} // namespace denox::compiler::cano

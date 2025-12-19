#pragma once

#include "compiler/cano/rules/IFusionRule.hpp"
#include "denox/algorithm/pattern_matching/EdgePattern.hpp"
#include "denox/algorithm/pattern_matching/NodePattern.hpp"
#include "model/ComputeOp.hpp"
#include "denox/symbolic/SymGraph.hpp"
#include <cassert>
namespace denox::compiler::cano {

class SliceSlice : public IFusionRule {
public:
  SliceSlice() {
    Pattern pattern;
    m_handles.A = m_handles.pattern.matchNode();
    m_handles.AB = m_handles.A->matchOutgoing();
    m_handles.B = m_handles.AB->matchDst();
    m_handles.BC = m_handles.B->matchOutgoing();
    m_handles.C = m_handles.BC->matchDst();

    m_handles.AB->matchRank(1);
    m_handles.BC->matchRank(1);

    m_handles.B->matchOutDeg(1);
    m_handles.B->matchInDeg(1);

    static auto isSliceOp = [](const ComputeOp &op) {
      return op.tag() == ComputeOpTag::Slice;
    };

    m_handles.AB->matchValue(isSliceOp);
    m_handles.BC->matchValue(isSliceOp);
  }

  const algorithm::GraphPattern<ComputeTensor, ComputeOp> &
  pattern() final override {
    return m_handles.pattern;
  }

  virtual void apply(SymGraph &symGraph,
                     const algorithm::LinkedGraphMatch<ComputeTensor, ComputeOp>
                         &match) final override {
    const auto &handles = m_handles;

    auto nodeA = match[handles.A];
    auto edgeAB = match[handles.AB];
    auto nodeC = match[handles.C];
    auto edgeBC = match[handles.BC];

    const auto &sliceAB = edgeAB.value().slice();
    const auto &sliceBC = edgeBC.value().slice();
    ComputeOpSlice sliceAC(
        symGraph.add(sliceAB->left, sliceBC->left),
        symGraph.min(sliceAB->right,
                     symGraph.add(sliceAB->left, sliceBC->right)),
        symGraph.add(sliceAB->top, sliceBC->top),
        symGraph.min(sliceAB->bottom,
                     symGraph.add(sliceAB->top, sliceBC->bottom)));

    nodeA->outgoing().insert_after(edgeAB.nextOutgoingIterator(), nodeC,
                                   sliceAC);
    edgeAB.erase();
  }

private:
  using Pattern = algorithm::GraphPattern<ComputeTensor, ComputeOp>;
  struct Handles {
    Pattern pattern;
    Pattern::NP A;
    Pattern::EP AB;
    Pattern::NP B;
    Pattern::EP BC;
    Pattern::NP C;
  };
  Handles m_handles;
};

} // namespace denox::compiler::cano

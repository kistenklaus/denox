#include "compiler/spec.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/small_vector.hpp"
#include "memory/container/vector.hpp"
#include "memory/tensor/ActivationLayout.hpp"
#include <exception>

namespace denox::compiler {

using Graph = CanoModel::Graph;
using NodeHandle = Graph::NodeHandle;

static inline void ensure_specialized_type(CanoModel::Graph::Node &node) {
  if (!node.value().type().has_value()) {
    node.value().setType(memory::Dtype::F16);
  }
}

static inline SpecModel::Graph::NodeHandle
create_spec_node(SpecModel &spec, const Lifetimes &lifetimes,
                 const CanoModel::Graph::NodeHandle &src,
                 const memory::ActivationLayout &layout) {
  const auto &ct = src->value();
  TensorInstance st{
      .extent = ct.extent(),
      .channels = ct.channels(),
      .layout = layout,
      .type = *ct.type(),
      .originalNode = src,
      .lifetime = lifetimes.valueLifetimes[*src->id()],
  };
  return spec.graph.createNode(std::move(st));
}

static inline memory::vector<SpecModel::Graph::NodeHandle> &ensure_variants_for(
    SpecModel &spec, const Lifetimes &lifetimes,
    memory::span<const memory::ActivationLayout> layouts,
    memory::vector<memory::vector<SpecModel::Graph::NodeHandle>> &variants,
    CanoModel::Graph::Node &n) {
  const auto id = n.id();
  if (static_cast<std::uint64_t>(id) >= variants.size())
    variants.resize(static_cast<std::uint64_t>(id) + 1);
  auto &bucket = variants[*id];
  if (!bucket.empty())
    return bucket;

  ensure_specialized_type(n);
  const auto &ct = n.value();

  if (ct.layout().has_value()) {
    bucket.push_back(create_spec_node(spec, lifetimes, n, *ct.layout()));
    return bucket;
  }

  const unsigned c = ct.channels();
  for (const auto &l : layouts) {
    if (l.supports(c))
      bucket.push_back(create_spec_node(spec, lifetimes, n, l));
  }
  if (bucket.empty())
    std::terminate(); // no supported layout
  return bucket;
}

SpecModel specialize(CanoModel &model, const Lifetimes &lifetimes,
                     memory::span<const memory::ActivationLayout> layouts) {
  assert(!layouts.empty());
  if (layouts.empty())
    std::terminate();

  using CanGraph = CanoModel::Graph;
  using SpecGraph = SpecModel::Graph;

  assert(model.input->value().layout().has_value());
  assert(model.output->value().layout().has_value());

  SpecModel spec{};

  memory::vector<memory::vector<SpecGraph::NodeHandle>> variants;
  variants.resize(model.graph.upperNodeCount());

  memory::dynamic_bitset seen(model.graph.upperNodeCount(), false);
  memory::vector<CanGraph::NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());
  stack.push_back(model.input);

  while (!stack.empty()) {
    CanGraph::NodeHandle node = std::move(stack.back());
    stack.pop_back();

    const auto nid = node->id();
    if (static_cast<std::uint64_t>(nid) >= seen.size())
      seen.resize(static_cast<std::uint64_t>(nid) + 1, false);
    if (seen[*nid])
      continue;
    seen[*nid] = true;

    ensure_variants_for(spec, lifetimes, layouts, variants, *node);

    for (const auto &e : node->outgoing()) {
      if (e.srcs().empty() || &e.srcs().front() != node.operator->()) {
        stack.emplace_back(e.dst());
        continue;
      }

      memory::vector<memory::vector<SpecGraph::NodeHandle> *> srcVarLists;
      srcVarLists.reserve(e.srcs().size());
      for (auto &s : e.srcs()) {
        auto &b = ensure_variants_for(spec, lifetimes, layouts, variants, s);
        srcVarLists.push_back(&b);
      }

      NodeHandle dstH = e.dst();
      auto &dstBucket =
          ensure_variants_for(spec, lifetimes, layouts, variants, *dstH);

      const std::size_t k = srcVarLists.size();
      if (k == 0) {
        stack.emplace_back(e.dst());
        continue;
      }

      memory::vector<std::size_t> idx(k, 0);
      bool done = false;
      while (!done) {
        SpecGraph::NodeHandle &anchor = (*srcVarLists[0])[idx[0]];

        memory::small_vector<const SpecGraph::NodeHandle *, 8> addSrcPtrs;
        addSrcPtrs.reserve(k - 1);
        for (std::size_t i = 1; i < k; ++i) {
          addSrcPtrs.push_back(&((*srcVarLists[i])[idx[i]]));
        }

        for (const auto &dstV : dstBucket) {
          anchor->outgoing().insert_after_with_dynamic_srcs(
              anchor->outgoing().begin(),
              memory::span<const SpecGraph::NodeHandle *>(addSrcPtrs), dstV,
              memory::NullWeight{}, e.value());
        }

        std::size_t pos = k;
        while (pos > 0) {
          --pos;
          ++idx[pos];
          if (idx[pos] < srcVarLists[pos]->size())
            break;
          idx[pos] = 0;
          if (pos == 0) {
            done = true;
            break;
          }
        }
      }

      stack.emplace_back(e.dst());
    }
  }

  {
    auto in = model.input;
    auto out = model.output;
    auto &inB = ensure_variants_for(spec, lifetimes, layouts, variants, *in);
    auto &outB = ensure_variants_for(spec, lifetimes, layouts, variants, *out);
    assert(inB.size() == 1 && "input must have a fixed layout");
    assert(outB.size() == 1 && "output must have a fixed layout");
    spec.input = inB.front();
    spec.output = outB.front();
  }

  return spec;
}

} // namespace denox::compiler

#include "denox/compiler/specialization/specialization.hpp"
#include "denox/common/TensorDataType.hpp"
#include "denox/common/TensorFormat.hpp"
#include "denox/common/TensorStorage.hpp"
#include "denox/compiler/lifeness/Lifetimes.hpp"
#include "denox/memory/container/dynamic_bitset.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/memory/hypergraph/NullWeight.hpp"

namespace denox::compiler {

static TensorFormat supported_ssbo_formats[] = {
    TensorFormat::SSBO_CHWC8,
    TensorFormat::SSBO_HWC,
};

static TensorFormat supported_tex_formats[] = {
    TensorFormat::TEX_RGBA,
};

static TensorStorage supported_tex_storages[] = {
    // TensorStorage::StorageImage,
    TensorStorage::SampledStorageImage,
};

struct TensorSpec {
  TensorFormat format;
  TensorStorage storage;
  TensorDataType dtype;
};

static std::vector<TensorSpec>
specialize_tensor(const CanoModel::Graph::NodeHandle &node) {
  std::vector<TensorStorage> storages;
  if (node->value().storage == TensorStorage::Optimal) {
    if (node->value().channels.isConstant() &&
        node->value().channels.constant() <= 4) {
      for (const auto &s : supported_tex_storages) {
        storages.push_back(s);
      }
    }
    storages.push_back(TensorStorage::StorageBuffer);
  } else {
    storages.push_back(node->value().storage);
  }

  std::vector<TensorFormat> formats;
  if (node->value().format == TensorFormat::Optimal) {
    if (node->value().channels.isConstant()) {
      const auto c = node->value().channels.constant();
      assert(c > 0);

      if (c <= 4) {
        if (std::ranges::count(supported_tex_formats, TensorFormat::TEX_RGBA) !=
            0) {
          formats.push_back(TensorFormat::TEX_RGBA);
        }
      }
      if (c <= 3) {
        if (std::ranges::count(supported_tex_formats, TensorFormat::TEX_RGB) !=
            0) {
          formats.push_back(TensorFormat::TEX_RGB);
        }
      }
      if (c <= 2) {
        if (std::ranges::count(supported_tex_formats, TensorFormat::TEX_RG) !=
            0) {
          formats.push_back(TensorFormat::TEX_RG);
        }
      }
      if (std::ranges::count(supported_tex_formats, TensorFormat::TEX_R) != 0) {
        formats.push_back(TensorFormat::TEX_R);
      }
    }
    for (const auto &f : supported_ssbo_formats) {
      formats.push_back(f);
    }

  } else {
    formats.push_back(node->value().format);
  }

  TensorDataType dtype = node->value().type == TensorDataType::Auto
                             ? TensorDataType::Float16
                             : node->value().type;

  std::vector<TensorSpec> out;

  for (TensorStorage s : storages) {
    for (TensorFormat f : formats) {

      const bool isTex =
          f == TensorFormat::TEX_R || f == TensorFormat::TEX_RG ||
          f == TensorFormat::TEX_RGB || f == TensorFormat::TEX_RGBA;

      const bool isBuffer = f == TensorFormat::SSBO_CHW ||
                            f == TensorFormat::SSBO_CHWC8 ||
                            f == TensorFormat::SSBO_HWC;

      if (isTex) {
        if (s != TensorStorage::StorageImage &&
            s != TensorStorage::SampledStorageImage)
          continue;
      }

      if (isBuffer) {
        if (s != TensorStorage::StorageBuffer)
          continue;
      }

      out.push_back(TensorSpec{
          .format = f,
          .storage = s,
          .dtype = dtype,
      });
    }
  }

  assert(!out.empty());
  return out;
}

static SpecModel::Graph::NodeHandle specialize_input(
    SpecModel &spec,
    memory::hash_map<uint64_t, memory::vector<SpecModel::Graph::NodeHandle>>
        &tensorSpecializations,
    const CanoModel::Graph::NodeHandle &input, const Lifetimes &lifetimes) {
  memory::vector<TensorSpec> specs = specialize_tensor(input);
  memory::vector<SpecModel::Graph::NodeHandle> specNodes;
  specNodes.reserve(specs.size());
  for (const auto &s : specs) {
    TensorInstance instance{
        .width = input->value().width,
        .height = input->value().height,
        .channels = input->value().channels,
        .storage = s.storage,
        .format = s.format,
        .type = s.dtype,
        .originalNode = input,
        .lifetime = lifetimes.valueLifetimes[*input->id()],
    };
    specNodes.push_back(spec.graph.createNode(std::move(instance)));
  }

  // Case 1: exactly one specialization → no dummy needed
  if (specNodes.size() == 1) {
    tensorSpecializations[*input->id()] = specNodes;
    return specNodes.front();
  }

  TensorInstance dummy{
      .width = input->value().width,
      .height = input->value().height,
      .channels = input->value().channels,
      .storage = TensorStorage::Optimal,
      .format = TensorFormat::Optimal,
      .type = TensorDataType::Float16,
      .originalNode = input,
      .lifetime = lifetimes.valueLifetimes[*input->id()],
  };
  auto dummyNode = spec.graph.createNode(std::move(dummy));

  for (const auto &dst : specNodes) {
    dummyNode->outgoing().insert(dst, ComputeOp{});
  }

  tensorSpecializations[*input->id()] = specNodes;
  return dummyNode;
}

SpecModel specialize(CanoModel &model, const Lifetimes &lifetimes) {
  SpecModel spec;

  memory::hash_map<std::uint64_t, std::vector<SpecModel::Graph::NodeHandle>>
      tensorSpecializations;

  spec.inputs.reserve(model.inputs.size());
  for (const auto &input : model.inputs) {
    spec.inputs.push_back(
        specialize_input(spec, tensorSpecializations, input, lifetimes));
  }

  // DFS: Specialize all reachable tensors.
  memory::dynamic_bitset visited(model.graph.upperNodeCount(), false);
  memory::vector<CanoModel::Graph::NodeHandle> stack;
  stack.reserve(model.graph.upperNodeCount());

  for (const auto &in : model.inputs) {
    stack.push_back(in);
  }

  while (!stack.empty()) {
    CanoModel::Graph::NodeHandle node = stack.back();
    stack.pop_back();

    const uint64_t nid = *node->id();
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;

    // Node itself must already be specialized if it is an input
    assert(tensorSpecializations.contains(nid));

    for (const auto &e : node->outgoing()) {
      const auto &dst = e.dst();
      const uint64_t did = *dst.id();

      if (!tensorSpecializations.contains(did)) {
        auto specs = specialize_tensor(dst);

        std::vector<SpecModel::Graph::NodeHandle> specNodes;
        specNodes.reserve(specs.size());

        for (const auto &s : specs) {
          TensorInstance instance{
              .width = dst.value().width,
              .height = dst.value().height,
              .channels = dst.value().channels,
              .storage = s.storage,
              .format = s.format,
              .type = s.dtype,
              .originalNode = dst,
              .lifetime = lifetimes.valueLifetimes[did],
          };
          specNodes.push_back(spec.graph.createNode(std::move(instance)));
        }

        tensorSpecializations[did] = std::move(specNodes);
      }

      stack.push_back(dst);
    }
  }

  // DFS: expand edges
  visited.clear();
  visited.resize(model.graph.upperNodeCount(), false);
  stack.clear();

  for (const auto &in : model.inputs) {
    stack.push_back(in);
  }

  while (!stack.empty()) {
    CanoModel::Graph::NodeHandle node = stack.back();
    stack.pop_back();

    const uint64_t nid = *node->id();
    if (visited[nid]) {
      continue;
    }
    visited[nid] = true;

    for (const auto &e : node->outgoing()) {
      const auto &dst = e.dst();
      std::vector<const std::vector<SpecModel::Graph::NodeHandle> *> srcLists;
      for (const auto &src : e.srcs()) {
        srcLists.push_back(&tensorSpecializations[*src.id()]);
      }

      auto &dstList = tensorSpecializations[*dst.id()];

      const size_t k = srcLists.size();
      assert(k > 0);

      std::vector<size_t> idx(k, 0);
      bool done = false;

      while (!done) {
        auto &anchor = (*srcLists[0])[idx[0]];

        memory::small_vector<const SpecModel::Graph::NodeHandle *, 4> extraSrcs;
        for (size_t i = 1; i < k; ++i) {
          extraSrcs.push_back(&(*srcLists[i])[idx[i]]);
        }

        for (auto &dstSpec : dstList) {
          anchor->outgoing().insert_after_with_dynamic_srcs(
              anchor->outgoing().begin(),
              memory::span<const SpecModel::Graph::NodeHandle *>(
                  extraSrcs.data(), extraSrcs.size()),
              dstSpec, memory::NullWeight{}, ComputeOp{e.value()});
        }

        // mixed-radix increment
        for (size_t p = k; p-- > 0;) {
          if (++idx[p] < srcLists[p]->size())
            break;
          idx[p] = 0;
          if (p == 0)
            done = true;
        }
      }

      stack.push_back(dst);
    }
  }

  spec.outputs.reserve(model.outputs.size());

  for (const auto &output : model.outputs) {
    const uint64_t oid = *output->id();
    assert(tensorSpecializations.contains(oid));

    auto &outSpecs = tensorSpecializations[oid];
    if (outSpecs.size() == 1) {
      spec.outputs.push_back(outSpecs.front());
      continue;
    }
    TensorInstance dummy{
        .width = output->value().width,
        .height = output->value().height,
        .channels = output->value().channels,
        .storage = TensorStorage::Optimal,
        .format = TensorFormat::Optimal,
        .type = TensorDataType::Float16,
        .originalNode = output,
        .lifetime = lifetimes.valueLifetimes[oid],
    };
    auto dummyNode = spec.graph.createNode(std::move(dummy));
    for (auto &src : outSpecs) {
      src->outgoing().insert(dummyNode, ComputeOp{});
    }
    spec.outputs.push_back(dummyNode);
  }
  return spec;
}

} // namespace denox::compiler

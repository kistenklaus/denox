#include "compiler/placement/placement.hpp"
#include "algorithm/align_up.hpp"
#include "compiler/ir/impl/MemoryConstrain.hpp"
#include "compiler/ir/impl/TensorId.hpp"
#include "compiler/ir/impl/TensorStorageRequirements.hpp"
#include "diag/invalid_state.hpp"
#include "diag/not_implemented.hpp"
#include "diag/unreachable.hpp"
#include "memory/container/dynamic_bitset.hpp"
#include "memory/container/vector.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

namespace denox::compiler {

// NOTE: must not be the same as alignof(std::uint32_t)
// some architectures are weird.
static constexpr std::size_t RO_SHADER_SRC_ALIGNMENT = 4;
static constexpr std::size_t RO_PARAM_ALIGNMENT = 8;

CompModel placement(const ImplModel &model) {
  CompModel compModel;
  compModel.symGraph = model.symGraph;
  SymGraph &symGraph = compModel.symGraph;

  // Precompute roSize.
  std::size_t roSize = 0;
  for (const auto &dispatch : model.dispatches) {
    roSize = algorithm::align_up(roSize, RO_SHADER_SRC_ALIGNMENT);
    roSize += dispatch.binary.spv.size() * sizeof(std::uint32_t);
  }
  for (const auto &param : model.parameters) {
    roSize = algorithm::align_up(roSize, RO_PARAM_ALIGNMENT);
    roSize += param.data.size();
  }
  compModel.roData.resize(roSize, std::byte('\0'));
  // Write shader source to roData
  std::size_t roOffset = 0;
  for (const auto &computeDispatch : model.dispatches) {
    roOffset = algorithm::align_up(roOffset, RO_SHADER_SRC_ALIGNMENT);

    std::memcpy(compModel.roData.data() + roOffset,
                computeDispatch.binary.spv.data(),
                computeDispatch.binary.spv.size() * sizeof(std::uint32_t));

    ShaderSourceView src{
        .offset = roOffset,
        .size = computeDispatch.binary.spv.size() * sizeof(std::uint32_t),
    };
    Dispatch dispatch{
        .src = src,
        .setBindings = {}, // <- handeled after tensor placement.
        .pushConstants = computeDispatch.pushConstants,
    };
    compModel.dispatches.push_back(std::move(dispatch));

    roOffset += computeDispatch.binary.spv.size() * sizeof(std::uint32_t);
  }

  static constexpr std::uint64_t u64sential =
      std::numeric_limits<std::uint64_t>::max();
  memory::vector<std::uint64_t> tensorIdToViewMap(model.tensors.size(),
                                                  u64sential);

  // Split into parameters and runtime tensors.
  memory::dynamic_bitset tensorIsParameter(model.tensors.size(), false);
  for (const Parameter &param : model.parameters) {
    tensorIsParameter[param.tensorId.index] = true;
  }
  memory::vector<std::uint64_t> runtimeTensorsIndicies; // tensor-ids.
  memory::vector<std::uint64_t> paramTensorIndicies;
  memory::vector<std::uint64_t> indexMapping(model.tensors.size());
  for (std::size_t tid = 0; tid < model.tensors.size(); ++tid) {
    if (tensorIsParameter[tid]) {
      indexMapping[tid] = paramTensorIndicies.size();
      paramTensorIndicies.emplace_back(tid);
    } else {
      indexMapping[tid] = runtimeTensorsIndicies.size();
      runtimeTensorsIndicies.emplace_back(tid);
    }
  }

  // Create buffers and views for parameters.
  for (std::size_t p = 0; p < paramTensorIndicies.size(); ++p) {
    const std::uint64_t tensorId = paramTensorIndicies[p];
    const TensorStorageRequirements &tensor = model.tensors[tensorId];
    Buffer buffer{
        .size = tensor.byteSize,
        .alignment = tensor.minAlignment,
        .initalizer = memory::nullopt,
    };
    const std::uint64_t bufferIndex = compModel.buffers.size();
    compModel.buffers.push_back(buffer);
    TensorView view{
        .buffer = bufferIndex,
        .offset = Sym::Const(0),
    };
    const std::uint64_t viewId = compModel.tensors.size();
    compModel.tensors.push_back(view);
    tensorIdToViewMap[tensorId] = viewId;

    if (tensor.meta != nullptr) {
      compModel.tensorInfo.push_back(*tensor.meta);
    } else {
      compModel.tensorInfo.push_back(memory::nullopt);
    }
  }

  // Create initalizers for parameters.
  for (const auto &param : model.parameters) {
    roOffset = algorithm::align_up(roOffset, RO_PARAM_ALIGNMENT);
    std::memcpy(compModel.roData.data() + roOffset, param.data.data(),
                param.data.size());
    std::uint64_t offset = roOffset;
    roOffset += param.data.size();

    const std::uint64_t tensorId = param.tensorId.index;
    const std::uint64_t viewId = tensorIdToViewMap[tensorId];
    const TensorView &view = compModel.tensors[viewId];
    Buffer &buffer = compModel.buffers[view.buffer];
    buffer.initalizer = offset;
  }

  // Create views for runtime tensors.
  // 1. Determine which runtime tensors have to life in the same
  //    block. Reduce to a linearly indexed array of blocks.
  memory::vector<std::uint64_t> unionFind(runtimeTensorsIndicies.size());
  for (std::size_t i = 0; i < unionFind.size(); ++i) {
    unionFind[i] = i;
  }
  auto find = [&](std::uint64_t x) {
    std::uint64_t n = x;
    do {
      x = n;
      n = unionFind[x];
    } while (x != n);
    return x;
  };
  for (const auto &constrain : model.memoryImplicitConcatConstrains) {
    const std::uint64_t t0 = constrain.src0.index;
    const std::uint64_t t1 = constrain.src1.index;
    const std::uint64_t t2 = constrain.dst.index;
    if (tensorIsParameter[t0] || tensorIsParameter[t1] ||
        tensorIsParameter[t2]) {
      diag::not_implemented();
    }

    const std::uint64_t i0 = indexMapping[t0];
    const std::uint64_t i1 = indexMapping[t1];
    const std::uint64_t i2 = indexMapping[t2];

    const std::uint64_t x0 = find(i0);
    const std::uint64_t x1 = find(i1);
    const std::uint64_t x2 = find(i2);

    unionFind[x0] = x0;
    unionFind[x1] = x0;
    unionFind[x2] = x0;
  }
  // Index compression.
  // Map union find result of find(indexMapping[tensorId])
  // to a linear index.
  memory::vector<std::uint64_t> blockIndicies(runtimeTensorsIndicies.size(),
                                              u64sential);
  std::uint64_t blockCount = 0;
  for (std::size_t t = 0; t < runtimeTensorsIndicies.size(); ++t) {
    std::uint64_t x0 = find(t);
    if (blockIndicies[x0] == u64sential) {
      blockIndicies[x0] = blockCount++;
    }
  }
  struct PlacementConstrain {
    std::uint64_t tensorId;
    Sym offset;
  };
  struct Block {
    memory::vector<std::uint64_t> tensorIds;
    memory::optional<Sym> size;
    memory::optional<unsigned int> alignment;
    memory::vector<PlacementConstrain> constrains;
  };
  memory::vector<Block> blocks(blockCount);
  for (std::size_t t = 0; t < runtimeTensorsIndicies.size(); ++t) {
    std::uint64_t x0 = find(t);
    std::size_t blockIndex = blockIndicies[x0];
    blocks[blockIndex].tensorIds.emplace_back(runtimeTensorsIndicies[t]);
  }

  for (const auto &constrain : model.memoryImplicitConcatConstrains) {
    const std::uint64_t t0 = constrain.src0.index;
    const std::uint64_t t1 = constrain.src1.index;
    const std::uint64_t t2 = constrain.dst.index;

    if (tensorIsParameter[t0] || tensorIsParameter[t1] ||
        tensorIsParameter[t2]) {
      diag::not_implemented();
    }

    const std::uint64_t x0 = find(indexMapping[t0]);
    const std::uint64_t x1 = find(indexMapping[t1]);
    const std::uint64_t x2 = find(indexMapping[t2]);
    assert(x0 == x1);
    assert(x1 == x2);

    std::size_t blockIndex = blockIndicies[x0];
    assert(std::ranges::count(blocks[blockIndex].tensorIds, x0) == 0);
    assert(std::ranges::count(blocks[blockIndex].tensorIds, x1) == 0);
    assert(std::ranges::count(blocks[blockIndex].tensorIds, x2) == 0);

    const auto &t0Tensor = model.tensors[t0];
    const auto &t2Tensor = model.tensors[t2];

    PlacementConstrain x0Placement;
    x0Placement.tensorId = t0;
    x0Placement.offset = Sym::Const(0);
    PlacementConstrain x1Placement;
    x1Placement.tensorId = t1;
    x1Placement.offset = t0Tensor.byteSize;
    PlacementConstrain x2Placement;
    x2Placement.tensorId = t2;
    x2Placement.offset = Sym::Const(0);

    auto &block = blocks[blockIndex];
    block.constrains.push_back(x0Placement);
    block.constrains.push_back(x1Placement);
    block.constrains.push_back(x2Placement);
    if (block.size.has_value()) {
      block.size = symGraph.max(*block.size, t2Tensor.byteSize);
    } else {
      block.size = t2Tensor.byteSize;
    }
    unsigned int alignment =
        std::max(t2Tensor.minAlignment, t0Tensor.minAlignment);
    if (block.alignment.has_value()) {
      block.alignment = std::max(*block.alignment, alignment);
    } else {
      block.alignment = alignment;
    }
  }

  memory::dynamic_bitset placed(model.tensors.size(), false);

  const std::uint64_t bufferOffset = compModel.buffers.size();
  for (std::size_t b = 0; b < blocks.size(); ++b) {

    const Block &block = blocks[b];

    // NOTE: Currently not the cleanest implementation just guards that
    // we do not forget to modify this if we later want to add more constrains.
    // Is only supposed to work with Implicit Concat constrains.
    assert(block.constrains.empty() || block.tensorIds.size() == 3);

    const std::uint64_t bufferIndex = b + bufferOffset;
    Sym size;
    // 1. determine block size and alignment
    if (block.size.has_value()) {
      size = *block.size;
    } else {
      // infer size.
      size = Sym::Const(0);
      for (std::size_t t = 0; t < block.tensorIds.size(); ++t) {
        const auto &tensor = model.tensors[t];
        size = symGraph.add(size, tensor.byteSize);
      }
    }
    unsigned int alignment;
    if (block.alignment.has_value()) {
      alignment = *block.alignment;
    } else {
      alignment = 1;
      for (std::size_t t = 0; t < block.tensorIds.size(); ++t) {
        const auto &tensor = model.tensors[t];
        alignment = std::max(alignment, tensor.minAlignment);
      }
    }

    Buffer buffer{
        .size = size,
        .alignment = alignment,
        .initalizer = memory::nullopt,
    };
    assert(compModel.buffers.size() == bufferIndex);
    compModel.buffers.push_back(buffer);

    for (const auto &constrain : block.constrains) {
      if (placed[constrain.tensorId]) {
        diag::unreachable();
      }
      std::uint64_t viewId = compModel.tensors.size();
      tensorIdToViewMap[constrain.tensorId] = viewId;
      TensorView view{
          .buffer = bufferIndex,
          .offset = constrain.offset,
      };
      compModel.tensors.push_back(view);
      placed[constrain.tensorId] = true;
    }
    for (std::uint64_t tensorId : block.tensorIds) {
      if (placed[tensorId]) {
        continue;
      }
      // NOTE: Not build for this!
      // Only handles the trivial case!
      // We could do more complex stuff here, but i think
      // this requires a relation aware symbolic engine, which
      // at this point we do not have.
      assert(block.constrains.empty());
      const std::uint64_t viewId = compModel.tensors.size();
      tensorIdToViewMap[tensorId] = viewId;
      TensorView view{
          .buffer = bufferIndex,
          .offset = Sym::Const(0),
      };
      compModel.tensors.push_back(view);
      placed[tensorId] = true;
    }
  }

  for (std::size_t d = 0; d < model.dispatches.size(); ++d) {
    const auto &computeDispatch = model.dispatches[d];
    auto &dispatch = compModel.dispatches[d];

    for (const TensorBinding &binding : computeDispatch.bindings) {
      auto setBindingIt =
          std::find_if(dispatch.setBindings.begin(), dispatch.setBindings.end(),
                       [&](const DescriptorSetBinding &setBinding) {
                         return setBinding.set == binding.set;
                       });
      if (setBindingIt == dispatch.setBindings.end()) {
        DescriptorSetBinding setBinding;
        setBinding.set = binding.set;
        dispatch.setBindings.push_back(setBinding);
        setBindingIt = --dispatch.setBindings.end();
      }
      assert(setBindingIt != dispatch.setBindings.end());

      auto bindingIt = std::find_if(
          setBindingIt->bindings.begin(), setBindingIt->bindings.end(),
          [&](const DescriptorBinding &descriptorBinding) {
            return descriptorBinding.binding == binding.binding;
          });
      if (bindingIt != setBindingIt->bindings.end()) {
        diag::invalid_state(); // <- Binding Collision.
      }
      std::uint64_t viewId = tensorIdToViewMap[binding.tensorId.index];
      DescriptorBinding descriptorBinding{};
      descriptorBinding.binding = binding.binding;
      descriptorBinding.access = binding.accessFlag;
      descriptorBinding.tensor = viewId;
      setBindingIt->bindings.push_back(descriptorBinding);
    }
  }

  for (std::size_t i = 0; i < model.inputs.size(); ++i) {
    InputDesc input = model.inputs[i];
    input.tensor.index = tensorIdToViewMap[input.tensor.index];
    compModel.inputs.push_back(input);
  }

  for (std::size_t o = 0; o < model.outputs.size(); ++o) {
    OutputDesc output = model.outputs[o];
    output.tensor.index = tensorIdToViewMap[output.tensor.index];
    compModel.outputs.push_back(output);
  }

  return compModel;
}

} // namespace denox::compiler

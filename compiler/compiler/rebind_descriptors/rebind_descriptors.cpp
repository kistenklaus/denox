#include "compiler/rebind_descriptors/rebind_descriptors.hpp"
#include "Options.hpp"
#include "compiler/ir/comp/CompModel.hpp"
#include "compiler/ir/impl/InputDesc.hpp"
#include "compiler/ir/impl/TensorBinding.hpp"
#include "compiler/ir/impl/TensorId.hpp"
#include "shaders/compiler/spirv_rebind.hpp"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <dnx.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <limits>

static constexpr size_t MAX_BOUND_DESCRIPTOR_SET = 32; // very high upper bound.
static constexpr size_t MAX_BINDING = 32;

static bool is_input(const denox::compiler::CompModel &compModel,
                     uint64_t tensor) {
  auto it = std::ranges::find_if(compModel.inputs,
                                 [&](const denox::compiler::InputDesc &input) {
                                   return input.tensor.index == tensor;
                                 });
  return it != compModel.inputs.end();
}

static bool is_output(const denox::compiler::CompModel &compModel,
                      uint64_t tensor) {
  auto it = std::ranges::find_if(
      compModel.outputs, [&](const denox::compiler::OutputDesc &output) {
        return output.tensor.index == tensor;
      });
  return it != compModel.outputs.end();
}

void denox::compiler::rebind_descriptors(CompModel &compModel,
                                         const Options &options) {

  std::vector<std::pair<bool, std::vector<SpirvDescriptorRebind>>>
      binaryRebinds(
          compModel.shaderBinaries.size(),
          std::make_pair<bool, std::vector<SpirvDescriptorRebind>>(false, {}));

  // look for dispatches which use the same binary.
  std::vector<bool> binaryVisited(compModel.shaderBinaries.size(), false);
  std::vector<uint32_t> originalSourceId(compModel.shaderBinaries.size(),
                                         std::numeric_limits<uint32_t>::max());
  std::vector<std::vector<uint32_t>> originalSpv;
  for (const Dispatch &dispatch : compModel.dispatches) {
    if (binaryVisited[dispatch.binaryId]) {
      uint32_t sourceId = static_cast<uint32_t>(originalSpv.size());
      originalSpv.push_back(compModel.shaderBinaries[dispatch.binaryId].spv);
      originalSourceId[dispatch.binaryId] = sourceId;
    }
    binaryVisited[dispatch.binaryId] = true;
  }

  const DescriptorPolicies &policies = options.descriptorPolicies;

  std::vector<bool> paramBitset(compModel.tensors.size(), false);
  for (const auto &initalizer : compModel.initializers) {
    paramBitset[initalizer.tensor] = true;
  }

  for (Dispatch &dispatch : compModel.dispatches) {
    uint32_t binaryId = dispatch.binaryId;
    std::vector<SpirvDescriptorRebind> rebinds;

    std::vector<uint32_t> binding_acc;
    binding_acc.resize(MAX_BOUND_DESCRIPTOR_SET);

    std::vector<bool> visited(MAX_BOUND_DESCRIPTOR_SET * MAX_BINDING, false);

    // assign inputs
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const bool isInput = is_input(compModel, tensorBinding.tensor);
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        if (!isInput || visited[key]) {
          continue;
        }
        visited[key] = true;
        uint32_t dstSet = policies.inputPolicy.set;
        uint32_t dstBinding = binding_acc[dstSet]++;

        if (srcSet != dstSet || srcBinding != dstBinding) {
          rebinds.push_back(SpirvDescriptorRebind{
              .src_set = static_cast<uint8_t>(srcSet),
              .src_binding = static_cast<uint8_t>(srcBinding),
              .dst_set = static_cast<uint8_t>(dstSet),
              .dst_binding = static_cast<uint8_t>(dstBinding)});
        }
      }
    }

    // assign outputs
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        const bool isOutput = is_output(compModel, tensorBinding.tensor);
        if (!isOutput || visited[key]) {
          continue;
        }
        visited[key] = true;

        uint32_t dstSet = policies.outputPolicy.set;
        uint32_t dstBinding = binding_acc[dstSet]++;

        if (srcSet != dstSet || srcBinding != dstBinding) {
          rebinds.push_back(SpirvDescriptorRebind{
              .src_set = static_cast<uint8_t>(srcSet),
              .src_binding = static_cast<uint8_t>(srcBinding),
              .dst_set = static_cast<uint8_t>(dstSet),
              .dst_binding = static_cast<uint8_t>(dstBinding)});
        }
      }
    }

    // assign reads
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        const bool isRead = tensorBinding.access == AccessFlag::ReadOnly ||
                            tensorBinding.access == AccessFlag::ReadWrite;
        const bool isParam = paramBitset[tensorBinding.tensor];
        if (!isRead || isParam || visited[key]) {
          continue;
        }
        visited[key] = true;

        uint32_t dstSet = policies.readPolicy.set;
        uint32_t dstBinding = binding_acc[dstSet]++;

        if (srcSet != dstSet || srcBinding != dstBinding) {
          rebinds.push_back(SpirvDescriptorRebind{
              .src_set = static_cast<uint8_t>(srcSet),
              .src_binding = static_cast<uint8_t>(srcBinding),
              .dst_set = static_cast<uint8_t>(dstSet),
              .dst_binding = static_cast<uint8_t>(dstBinding)});
        }
      }
    }

    // assign writes
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        const bool isWrite = tensorBinding.access == AccessFlag::WriteOnly ||
                             tensorBinding.access == AccessFlag::ReadWrite;
        if (!isWrite || visited[key]) {
          continue;
        }
        visited[key] = true;

        uint32_t dstSet = policies.writePolicy.set;
        uint32_t dstBinding = binding_acc[dstSet]++;

        if (srcSet != dstSet || srcBinding != dstBinding) {
          rebinds.push_back(SpirvDescriptorRebind{
              .src_set = static_cast<uint8_t>(srcSet),
              .src_binding = static_cast<uint8_t>(srcBinding),
              .dst_set = static_cast<uint8_t>(dstSet),
              .dst_binding = static_cast<uint8_t>(dstBinding)});
        }
      }
    }

    // assign params
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        const bool isParam = paramBitset[tensorBinding.tensor];
        if (!isParam || visited[key]) {
          continue;
        }
        visited[key] = true;

        uint32_t dstSet = policies.paramPolicy.set;
        uint32_t dstBinding = binding_acc[dstSet]++;

        if (srcSet != dstSet || srcBinding != dstBinding) {
          rebinds.push_back(SpirvDescriptorRebind{
              .src_set = static_cast<uint8_t>(srcSet),
              .src_binding = static_cast<uint8_t>(srcBinding),
              .dst_set = static_cast<uint8_t>(dstSet),
              .dst_binding = static_cast<uint8_t>(dstBinding)});
        }
      }
    }
    // sanity check
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        const uint32_t srcSet = setBinding.set;
        const uint32_t srcBinding = tensorBinding.binding;
        const uint32_t key = srcSet * MAX_BINDING + srcBinding;
        assert(visited[key]);
      }
    }

    // rebind within SPIR-V binaries.
    if (binaryRebinds[binaryId].first == false) {
      spirv_rebind_descriptors(compModel.shaderBinaries[binaryId].spv, rebinds);
      binaryRebinds[binaryId].first = true;
      binaryRebinds[binaryId].second = rebinds;
    } else {
      const auto &previous_rebinds = binaryRebinds[binaryId].second;
      bool equivalent = true;
      for (const SpirvDescriptorRebind &rebind : rebinds) {
        bool match_found = false;
        for (const SpirvDescriptorRebind &prev_rebind : previous_rebinds) {
          if (rebind.src_set == prev_rebind.src_set &&
              rebind.src_binding == prev_rebind.src_binding &&
              rebind.dst_set == prev_rebind.dst_set &&
              rebind.dst_binding == prev_rebind.dst_binding) {
            match_found = true;
            break;
          }
        }
        if (!match_found) {
          equivalent = false;
          break;
        }
      }
      if (equivalent) {
        // there we do not have to do anything to the spirv-binary.
      } else {
        // We found a case, where the rebinding of the descriptor sets made two
        // previously identical binaries different.
        const std::vector<uint32_t> &original_spv =
            originalSpv[originalSourceId[binaryId]];
        std::vector<uint32_t> spv = original_spv;
        spirv_rebind_descriptors(spv, rebinds);
        binaryId = static_cast<uint32_t>(compModel.shaderBinaries.size());
        dispatch.binaryId = binaryId;
        compModel.shaderBinaries.emplace_back(ShaderBinary{spv});
      }
    }

    // Rebind tensor bindings.
    struct FlatTensorBinding {
      uint32_t set;
      uint32_t binding;
      AccessFlag access;
      uint64_t tensor;
    };
    std::vector<FlatTensorBinding> bindings;
    for (const auto &setBinding : dispatch.setBindings) {
      for (const auto &tensorBinding : setBinding.bindings) {
        bindings.push_back(FlatTensorBinding{
            .set = setBinding.set,
            .binding = tensorBinding.binding,
            .access = tensorBinding.access,
            .tensor = tensorBinding.tensor,
        });
      }
    }

    // apply rebinds.
    std::vector<bool> rebound(bindings.size(), false);
    for (const auto &rebind : rebinds) {
      bool found = false;
      for (size_t i = 0; i < bindings.size(); ++i) {
        if (rebound[i]) {
          continue;
        }
        auto &binding = bindings[i];
        if (binding.set == rebind.src_set &&
            binding.binding == rebind.src_binding) {
          rebound[i] = true;
          binding.set = rebind.dst_set;
          binding.binding = rebind.dst_binding;
          found = true;
          break;
        }
      }
      assert(found);
    }

    std::ranges::sort(bindings, [](const FlatTensorBinding &lhs,
                                   const FlatTensorBinding &rhs) {
      if (lhs.set != rhs.set)
        return lhs.set < rhs.set;
      return lhs.binding < rhs.binding;
    });

    dispatch.setBindings.clear();
    for (const auto &binding : bindings) {
      auto setIt = std::ranges::find_if(dispatch.setBindings,
                                        [&](const auto &setBinding) {
                                          return setBinding.set == binding.set;
                                        });
      if (setIt == dispatch.setBindings.end()) {
        dispatch.setBindings.push_back(DescriptorSetBinding{
            .set = binding.set,
            .bindings = {},
        });
        setIt = dispatch.setBindings.end() - 1;
      }
      // checks for duplicate bindings.
      auto bindingIt =
          std::ranges::find_if(setIt->bindings, [&](const auto &tensorBinding) {
            return tensorBinding.binding == binding.binding;
          });
      assert(bindingIt == setIt->bindings.end());
      setIt->bindings.push_back(DescriptorBinding{
          .binding = binding.binding,
          .access = binding.access,
          .tensor = binding.tensor,
      });
    }
  }
}

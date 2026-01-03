#include "denox/compiler/rebind_descriptors/rebind_descriptors.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <algorithm>
#include <dnx.h>
#include <fmt/format.h>
#include <limits>

static constexpr size_t MAX_BOUND_DESCRIPTOR_SET = 32; // very high upper bound.
static constexpr size_t MAX_BINDING = 32;

static bool is_input(const denox::compiler::SpvSchedule &schedule,
                     denox::compiler::TensorId tensorId) {
  return std::ranges::count(schedule.inputs, tensorId.index) != 0;
}

static bool is_output(const denox::compiler::SpvSchedule &schedule,
                      denox::compiler::TensorId tensorId) {
  return std::ranges::count(schedule.outputs, tensorId.index) != 0;
}

void denox::compiler::rebind_descriptors(SpvSchedule &schedule,
                                         const CompileOptions &options,
                                         spirv::SpirvTools *spirvTools) {

  std::vector<std::pair<bool, std::vector<spirv::SpirvDescriptorRebind>>>
      binaryRebinds(
          schedule.binaries.size(),
          std::make_pair<bool, std::vector<spirv::SpirvDescriptorRebind>>(false,
                                                                          {}));

  // look for dispatches which use the same binary.
  std::vector<bool> binaryVisited(schedule.binaries.size(), false);
  std::vector<uint32_t> originalSourceId(schedule.binaries.size(),
                                         std::numeric_limits<uint32_t>::max());
  std::vector<std::vector<uint32_t>> originalSpv;
  for (const auto &dispatch : schedule.dispatches) {
    if (binaryVisited[dispatch.binaryId]) {
      uint32_t sourceId = static_cast<uint32_t>(originalSpv.size());
      originalSpv.push_back(schedule.binaries[dispatch.binaryId].spv);
      originalSourceId[dispatch.binaryId] = sourceId;
    }
    binaryVisited[dispatch.binaryId] = true;
  }

  const DescriptorPolicies &policies = options.descriptorPolicies;

  std::vector<bool> paramBitset(schedule.tensors.size(), false);
  for (const auto &initializer : schedule.initializers) {
    paramBitset[static_cast<size_t>(initializer.tensor)] = true;
  }

  for (auto &dispatch : schedule.dispatches) {
    // Make descriptor packing deterministic across runs.
    std::ranges::sort(dispatch.bindings,
                      [](const TensorBinding &lhs, const TensorBinding &rhs) {
                        if (lhs.set != rhs.set)
                          return lhs.set < rhs.set;
                        return lhs.binding < rhs.binding;
                      });

    // Enforce current assumption: no ReadWrite bindings yet.
    for (const auto &b : dispatch.bindings) {
      assert(b.accessFlag != Access::ReadWrite &&
             "Access::ReadWrite is not supported yet (rebind_descriptors "
             "assumes RO/WO only)");
      // Optional safety if you keep MAX_* fixed:
      assert(b.set < MAX_BOUND_DESCRIPTOR_SET);
      assert(b.binding < MAX_BINDING);
    }

    uint32_t binaryId = dispatch.binaryId;
    std::vector<spirv::SpirvDescriptorRebind> rebinds;

    std::vector<uint32_t> binding_acc(MAX_BOUND_DESCRIPTOR_SET, 0u);
    std::vector<bool> visited(MAX_BOUND_DESCRIPTOR_SET * MAX_BINDING, false);

    // assign inputs
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const bool isInput = is_input(schedule, tensorBinding.tensorId);
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      if (!isInput || visited[key])
        continue;
      visited[key] = true;

      const uint32_t dstSet = policies.inputPolicy.set;
      const uint32_t dstBinding = binding_acc[dstSet]++;

      if (srcSet != dstSet || srcBinding != dstBinding) {
        rebinds.push_back({.src_set = static_cast<uint8_t>(srcSet),
                           .src_binding = static_cast<uint8_t>(srcBinding),
                           .dst_set = static_cast<uint8_t>(dstSet),
                           .dst_binding = static_cast<uint8_t>(dstBinding)});
      }
    }

    // assign outputs
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      const bool isOutput = is_output(schedule, tensorBinding.tensorId);
      if (!isOutput || visited[key])
        continue;
      visited[key] = true;

      const uint32_t dstSet = policies.outputPolicy.set;
      const uint32_t dstBinding = binding_acc[dstSet]++;

      if (srcSet != dstSet || srcBinding != dstBinding) {
        rebinds.push_back({.src_set = static_cast<uint8_t>(srcSet),
                           .src_binding = static_cast<uint8_t>(srcBinding),
                           .dst_set = static_cast<uint8_t>(dstSet),
                           .dst_binding = static_cast<uint8_t>(dstBinding)});
      }
    }

    // assign reads (RO only; RW asserted impossible above)
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      const bool isRead = (tensorBinding.accessFlag == Access::ReadOnly);
      const bool isParam = paramBitset[tensorBinding.tensorId.index];
      if (!isRead || isParam || visited[key])
        continue;
      visited[key] = true;

      const uint32_t dstSet = policies.readPolicy.set;
      const uint32_t dstBinding = binding_acc[dstSet]++;

      if (srcSet != dstSet || srcBinding != dstBinding) {
        rebinds.push_back({.src_set = static_cast<uint8_t>(srcSet),
                           .src_binding = static_cast<uint8_t>(srcBinding),
                           .dst_set = static_cast<uint8_t>(dstSet),
                           .dst_binding = static_cast<uint8_t>(dstBinding)});
      }
    }

    // assign writes (WO only; RW asserted impossible above)
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      const bool isWrite = (tensorBinding.accessFlag == Access::WriteOnly);
      if (!isWrite || visited[key])
        continue;
      visited[key] = true;

      const uint32_t dstSet = policies.writePolicy.set;
      const uint32_t dstBinding = binding_acc[dstSet]++;

      if (srcSet != dstSet || srcBinding != dstBinding) {
        rebinds.push_back({.src_set = static_cast<uint8_t>(srcSet),
                           .src_binding = static_cast<uint8_t>(srcBinding),
                           .dst_set = static_cast<uint8_t>(dstSet),
                           .dst_binding = static_cast<uint8_t>(dstBinding)});
      }
    }

    // assign params
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      const bool isParam = paramBitset[tensorBinding.tensorId.index];
      if (!isParam || visited[key])
        continue;
      visited[key] = true;

      const uint32_t dstSet = policies.paramPolicy.set;
      const uint32_t dstBinding = binding_acc[dstSet]++;

      if (srcSet != dstSet || srcBinding != dstBinding) {
        rebinds.push_back({.src_set = static_cast<uint8_t>(srcSet),
                           .src_binding = static_cast<uint8_t>(srcBinding),
                           .dst_set = static_cast<uint8_t>(dstSet),
                           .dst_binding = static_cast<uint8_t>(dstBinding)});
      }
    }

    // sanity check
    for (const auto &tensorBinding : dispatch.bindings) {
      const uint32_t srcSet = tensorBinding.set;
      const uint32_t srcBinding = tensorBinding.binding;
      const uint32_t key = srcSet * MAX_BINDING + srcBinding;
      assert(visited[key]);
    }

    // rebind within SPIR-V binaries.
    if (binaryRebinds[binaryId].first == false) {
      spirvTools->rebind(schedule.binaries[binaryId], rebinds);
      binaryRebinds[binaryId].first = true;
      binaryRebinds[binaryId].second = rebinds;
    } else {
      const auto &previous_rebinds = binaryRebinds[binaryId].second;
      bool equivalent = true;
      for (const spirv::SpirvDescriptorRebind &rebind : rebinds) {
        bool match_found = false;
        for (const spirv::SpirvDescriptorRebind &prev_rebind :
             previous_rebinds) {
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
        SpirvBinary binary{spv};
        spirvTools->rebind(binary, rebinds);
        binaryId = static_cast<uint32_t>(schedule.binaries.size());
        dispatch.binaryId = binaryId;
        schedule.binaries.emplace_back(binary);
      }
    }

    // apply rebinds to tensor bindings (flat array)
    std::vector<bool> rebound(dispatch.bindings.size(), false);
    for (const auto &rebind : rebinds) {
      bool found = false;
      for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
        if (rebound[i]) {
          continue;
        }
        auto &binding = dispatch.bindings[i];
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

    std::ranges::sort(dispatch.bindings,
                      [](const TensorBinding &lhs, const TensorBinding &rhs) {
                        if (lhs.set != rhs.set)
                          return lhs.set < rhs.set;
                        return lhs.binding < rhs.binding;
                      });

    // checks for duplicate bindings (same set+binding)
    for (size_t i = 1; i < dispatch.bindings.size(); ++i) {
      const auto &a = dispatch.bindings[i - 1];
      const auto &b = dispatch.bindings[i];
      assert(!(a.set == b.set && a.binding == b.binding));
    }
  }
}

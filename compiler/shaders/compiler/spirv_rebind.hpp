#pragma once

#include <cstdint>
#include <span>

namespace denox::compiler {

struct SpirvDescriptorRebind {
  uint8_t src_set;
  uint8_t src_binding;
  uint8_t dst_set;
  uint8_t dst_binding;
};

void spirv_rebind_descriptors(std::span<uint32_t> spirv,
                              std::span<const SpirvDescriptorRebind> rebind);
} // namespace denox::compiler

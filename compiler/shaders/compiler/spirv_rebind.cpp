#include "shaders/compiler/spirv_rebind.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

static constexpr uint16_t OpDecorate = 71;
static constexpr uint32_t MagicNumber = 0x07230203;
static constexpr uint32_t DecorationDescriptorSet = 34;
static constexpr uint32_t DecorationBinding = 33;

void denox::compiler::spirv_rebind_descriptors(
    std::span<uint32_t> spirv, std::span<const SpirvDescriptorRebind> rebinds) {
  assert(spirv.size() > 5);

  const uint32_t magic = spirv[0];
  assert(magic == MagicNumber);

  uint32_t pc = 5;
  struct RebindLoc {
    uint32_t id;
    uint32_t pc_set;
    uint32_t pc_binding;
  };
  std::vector<RebindLoc> locs;
  locs.reserve(10);

  const auto try_rebind_set = [&](uint32_t id, uint32_t pc_set,
                                  uint32_t pc_binding) {
    assert(pc_set != 0 || pc_binding != 0);
    auto loc = std::ranges::find_if(
        locs, [id](const RebindLoc &loc) { return loc.id == id; });
    if (loc == locs.end()) {
      locs.emplace_back(id, pc_set, pc_binding);
      return;
    }
    loc->pc_binding |= pc_binding;
    loc->pc_set |= pc_set;
    if (loc->pc_set != 0 && loc->pc_binding != 0) {
      const uint32_t set = spirv[loc->pc_set];
      const uint32_t binding = spirv[loc->pc_binding];
      auto rebind = std::ranges::find_if(
          rebinds, [set, binding](const SpirvDescriptorRebind &rebind) {
            return rebind.src_set == set && rebind.src_binding == binding;
          });
      if (rebind != rebinds.end()) {
        spirv[loc->pc_set] = rebind->dst_set;
        spirv[loc->pc_binding] = rebind->dst_binding;
      }
      std::swap(*loc, locs.back());
      locs.pop_back();
    }
  };

  while (pc < spirv.size()) {
    uint32_t op = spirv[pc];
    uint32_t size = op >> 16;
    uint32_t opcode = op & 0xFFFF;
    assert(size > 0);
    assert(pc + size <= spirv.size());
    if (opcode == OpDecorate) {
      assert(size >= 3);
      const uint32_t id = spirv[pc + 1];
      const uint32_t decoration = spirv[pc + 2];
      if (decoration == DecorationDescriptorSet) {
        assert(size >= 4);
        try_rebind_set(id, pc + 3, 0);
      } else if (decoration == DecorationBinding) {
        assert(size >= 4);
        try_rebind_set(id, 0, pc + 3);
      }
    }
    pc += size;
  }
}

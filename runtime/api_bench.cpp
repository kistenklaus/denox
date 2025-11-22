
#include "context.hpp"
#include "denox/runtime.hpp"
#include "instance.hpp"
#include "vma.hpp"
#include <cmath>
#include <fmt/base.h>
#include <iterator>
#include <variant>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace denox {

int bench_runtime_instance(RuntimeContext context, RuntimeInstance instance) {
  auto *ctx = static_cast<runtime::Context *>(context);
  const auto *mi = static_cast<runtime::Instance *>(instance);

  uint32_t queryCount = mi->cmds.size() * 2;
  VkQueryPool queryPool = ctx->createTimestampQueryPool(queryCount);

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

  ctx->cmdResetQueryPool(cmd, queryPool, 0, queryCount);

  struct TimingResult {
    std::string input_desc;
    std::string output_desc;
    std::string name;
    uint32_t startTimestamp;
    uint32_t endTimestamp;
    size_t memory_accesses;
  };
  std::vector<TimingResult> timings;

  uint32_t previousTimestamp = 0;
  ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                         previousTimestamp);

  for (std::size_t pc = 0; pc < mi->cmds.size(); ++pc) {
    const auto &op = mi->cmds[pc];
    if (std::holds_alternative<runtime::InstanceDispatch>(op)) {
      const auto &dispatch = std::get<runtime::InstanceDispatch>(op);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        dispatch.dispatch->pipeline);

      vkCmdPushConstants(cmd, dispatch.dispatch->pipelineLayout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         dispatch.dispatch->dispatch->push_constant()->size(),
                         dispatch.pushConstantValues);

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              dispatch.dispatch->pipelineLayout, 0,
                              dispatch.dispatch->descriptorSets.size(),
                              dispatch.descriptorSets, 0, nullptr);

      vkCmdDispatch(cmd, dispatch.workgroupCounts[0],
                    dispatch.workgroupCounts[1], dispatch.workgroupCounts[2]);

      uint32_t start = previousTimestamp;
      uint32_t end = ++previousTimestamp;
      ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                             end);

      fmt::println("debug = {}", dispatch.memory_writes.value_or(0));

      timings.push_back(TimingResult{
          .input_desc = dispatch.input_desc.value_or(""),
          .output_desc = dispatch.output_desc.value_or(""),
          .name = dispatch.name.value_or(fmt::format("operation-{}", pc)),
          .startTimestamp = start,
          .endTimestamp = end,
          .memory_accesses = dispatch.memory_reads.value_or(0) +
                             dispatch.memory_writes.value_or(0),
      });

    } else if (std::holds_alternative<runtime::InstanceBarrier>(op)) {
      const auto &barrier = std::get<runtime::InstanceBarrier>(op);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           barrier.bufferBarrier.size(),
                           barrier.bufferBarrier.data(), 0, nullptr);
    }
  }

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

  ++previousTimestamp;
  ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                         previousTimestamp);
  ++previousTimestamp;

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  std::vector<uint64_t> timestamps =
      ctx->getQueryResults(queryPool, previousTimestamp);

  ctx->destroyQueryPool(queryPool);

  size_t totalAccesses = 0;
  for (size_t i = 0; i < timings.size(); ++i) {
    const auto &t = timings[i];
    float duration =
        ctx->timestampDifference(timestamps, t.startTimestamp, t.endTimestamp);

    auto s = fmt::format("{:.3f}", duration); // e.g. "123.456"
    auto pos = s.find('.');
    auto int_part = s.substr(0, pos);
    auto frac_part = s.substr(pos + 1);
    constexpr int W_INT = 6;
    constexpr int W_FRAC = 3;

    float throughput =
        (static_cast<float>(t.memory_accesses) / (duration * 1e-3)) * 1e-9;
    totalAccesses += t.memory_accesses;

    fmt::println("{:>22} \x1B[34m{:-^40}>\x1B[0m {:<22} :{:>3}.{:<3}ms "
                 "\x1B[90m({:>4} GB/s)\x1B[0m    {:.3f}MB",
                 t.input_desc, t.name, t.output_desc, int_part, frac_part,
                 static_cast<uint32_t>(std::round(throughput)), t.memory_accesses * 1e-6);
  }
  float totalDuration =
      ctx->timestampDifference(timestamps, 0, previousTimestamp - 1);
  fmt::println("\x1B[31m{:>89} {:.3f}ms\x1B[0m \x1B[90m({:>4} GB/s)\x1B[0m",
               "Total time :", totalDuration, (totalAccesses / (totalDuration*1e-3)) * 1e-9);

  return 0;
}

} // namespace denox

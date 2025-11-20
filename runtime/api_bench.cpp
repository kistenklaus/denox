
#include "context.hpp"
#include "instance.hpp"
#include "denox/runtime.hpp"
#include "vma.hpp"
#include <fmt/base.h>
#include <variant>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace denox {

int bench_runtime_instance(RuntimeContext context, RuntimeInstance instance) {
  auto* ctx = static_cast<runtime::Context*>(context);
  const auto* mi = static_cast<runtime::Instance*>(instance);

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);


  for (std::size_t pc = 0; pc < mi->cmds.size(); ++pc) {
    const auto &op = mi->cmds[pc];
    if (std::holds_alternative<runtime::InstanceDispatch>(op)) {
      const auto &dispatch = std::get<runtime::InstanceDispatch>(op);

      fmt::println("{}", dispatch.debug_info.value());

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

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  return 0;
}

}

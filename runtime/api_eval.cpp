#include "context.hpp"
#include "denox/runtime.hpp"
#include "instance.hpp"
#include "vma.hpp"
#include <variant>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace denox {

int eval_runtime_instance(RuntimeContext context, RuntimeInstance instance,
                          const void **inputs, void **outputs) {
  auto *ctx = static_cast<runtime::Context *>(context);
  const auto *mi = static_cast<runtime::Instance *>(instance);

  // Create staging buffers.
  std::vector<runtime::Buffer> inputStages(mi->inputs.size());
  for (std::size_t i = 0; i < mi->inputs.size(); ++i) {
    const auto &input = mi->inputs[i];
    const auto &tensor = mi->tensors[input.tensor];
    runtime::Buffer runtimeBuffer = ctx->createBuffer(
        tensor.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    ctx->copy(runtimeBuffer.allocation, inputs[i], tensor.size);

    inputStages[i] = runtimeBuffer;
  }
  std::vector<runtime::Buffer> outputStages(mi->outputs.size());
  for (std::size_t o = 0; o < mi->outputs.size(); ++o) {
    const auto &output = mi->outputs[o];
    const auto &tensor = mi->tensors[output.tensor];
    outputStages[o] =
        ctx->createBuffer(tensor.size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
  }

  VkCommandPool cmdPool = ctx->createCommandPool();
  VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);

  // global barrier to wait for host writes.
  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
  for (std::size_t i = 0; i < mi->inputs.size(); ++i) {
    const auto &input = mi->inputs[i];
    const auto &tensor = mi->tensors[input.tensor];
    const auto &buffer = mi->buffers[tensor.buffer];
    const auto &stage = inputStages[i];
    ctx->cmdCopy(cmd, buffer.buffer, stage, tensor.size, tensor.offset, 0);
  }

  // global barrier to wait for all transfers to be complete.
  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

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
    } else if (std::holds_alternative<runtime::InstanceBarrier>(op)) {
      const auto &barrier = std::get<runtime::InstanceBarrier>(op);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                           barrier.bufferBarrier.size(),
                           barrier.bufferBarrier.data(), 0, nullptr);
    }
  }

  // Wait for all compute shaders to be done with writing.
  ctx->cmdMemoryBarrier(
      cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

  for (std::size_t o = 0; o < mi->outputs.size(); ++o) {
    const auto &output = mi->outputs[o];
    const auto &tensor = mi->tensors[output.tensor];
    const auto &buffer = mi->buffers[tensor.buffer];
    const auto &stage = outputStages[o];
    ctx->cmdCopy(cmd, stage, buffer.buffer, tensor.size, 0, tensor.offset);
  }

  ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

  ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
  ctx->destroyCommandPool(cmdPool);

  // Download to from stage to host.
  for (std::size_t o = 0; o < mi->outputs.size(); ++o) {
    const auto &output = mi->outputs[o];
    const auto &tensor = mi->tensors[output.tensor];
    ctx->copy(outputs[o], outputStages[o].allocation, tensor.size);
  }

  // Cleanup temporary buffers.
  for (runtime::Buffer &buffer : outputStages) {
    ctx->destroyBuffer(buffer);
  }
  for (runtime::Buffer &buffer : inputStages) {
    ctx->destroyBuffer(buffer);
  }

  return 0;
}

} // namespace denox

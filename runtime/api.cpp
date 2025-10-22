#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "vma.hpp"
#include <algorithm>
#include <fmt/printf.h>
#include <stdexcept>
#include <type_traits>
#include <vulkan/vulkan_core.h>

namespace denox {


//
// int create_runtime_model_instance(RuntimeContext context, RuntimeModel model,
//                                   int dynamicExtentCount,
//                                   DynamicExtent *dynamicExtents,
//                                   RuntimeModelInstance *instance) {
//   assert(context != nullptr);
//   assert(model != nullptr);
//   const auto m = reinterpret_cast<runtime::Model *>(model);
//   auto ctx = reinterpret_cast<runtime::Context *>(context);
//
//   auto mi = new runtime::ModelInstance();
//   mi->model = m;
//   { // Evaluate all symbolic expressions.
//     const dnx::SymIR *ir = m->dnx->sym_ir();
//     std::size_t dpsize = static_cast<std::size_t>(ir->var_count()) +
//                          static_cast<std::size_t>(ir->ops()->size());
//     mi->vars.resize(dpsize);
//     for (std::size_t i = 0; i < dynamicExtentCount; ++i) {
//       const DynamicExtent &extent = dynamicExtents[i];
//       auto it = std::find_if(
//           m->dnx->value_names()->begin(), m->dnx->value_names()->end(),
//           [&extent](const dnx::ValueName *valueName) {
//             return std::strcmp(valueName->name()->c_str(), extent.name) == 0;
//           });
//       if (it == m->dnx->value_names()->end()) {
//         throw std::runtime_error(
//             fmt::format("Dynamic extent {} does not exist.", extent.name));
//       }
//       const dnx::ValueName *valueName = *it;
//       if (valueName->value_type() == dnx::ScalarSource_literal) {
//         throw std::runtime_error(
//             fmt::format("Dynamic extent {} is not dynamic.", extent.name));
//       }
//       const dnx::SymRef *symRef = valueName->value_as_symbolic();
//       if (symRef->sid() >= ir->var_count()) {
//         throw std::runtime_error(fmt::format(
//             "Dynamic extent {} is not a dynamic variable.", extent.name));
//       }
//       std::uint32_t sid = symRef->sid();
//       mi->vars[sid] = static_cast<std::int64_t>(extent.value);
//     }
//     for (std::uint64_t pc = 0; pc < ir->ops()->size(); ++pc) {
//       std::uint64_t rx = pc + ir->var_count();
//       const dnx::SymIROp *op = ir->ops()->Get(pc);
//       std::int64_t lhs;
//       if (op->opcode() & dnx::SymIROpCode_LHSC) {
//         lhs = op->lhs();
//       } else {
//         lhs = mi->vars[static_cast<std::uint64_t>(op->lhs())];
//       }
//       std::int64_t rhs;
//       if (op->opcode() & dnx::SymIROpCode_RHSC) {
//         rhs = op->rhs();
//       } else {
//         rhs = mi->vars[static_cast<std::uint64_t>(op->rhs())];
//       }
//       std::underlying_type_t<dnx::SymIROpCode> opcode =
//           op->opcode() & ~(dnx::SymIROpCode_LHSC | dnx::SymIROpCode_RHSC);
//
//       switch (opcode) {
//       case dnx::SymIROpCode_NOP:
//         break;
//       case dnx::SymIROpCode_ADD:
//         mi->vars[rx] = lhs + rhs;
//         break;
//       case dnx::SymIROpCode_SUB:
//         mi->vars[rx] = lhs - rhs;
//         break;
//       case dnx::SymIROpCode_MUL:
//         mi->vars[rx] = lhs * rhs;
//         break;
//       case dnx::SymIROpCode_DIV:
//         mi->vars[rx] = lhs / rhs;
//         break;
//       case dnx::SymIROpCode_MOD:
//         mi->vars[rx] = (lhs % rhs);
//         if (mi->vars[rx] < 0) {
//           mi->vars[rx] += rhs;
//         }
//         break;
//       case dnx::SymIROpCode_MIN:
//         mi->vars[rx] = std::min(lhs, rhs);
//         break;
//       case dnx::SymIROpCode_MAX:
//         mi->vars[rx] = std::max(lhs, rhs);
//         break;
//       default:
//         throw std::runtime_error("Invalid sym instruction.");
//       }
//     }
//   }
//
//   { // Create (get) buffers.
//     mi->buffers.resize(m->dnx->buffers()->size(),
//                        runtime::Buffer{
//                            .buffer = VK_NULL_HANDLE,
//                            .allocation = VK_NULL_HANDLE,
//                        });
//     mi->ownedBuffers.resize(m->dnx->buffers()->size(), false);
//     // Setup initalizers.
//     for (std::size_t i = 0; i < m->dnx->initializers()->size(); ++i) {
//       const dnx::BufferInitializer *init = m->dnx->initializers()->Get(i);
//       mi->buffers[init->buffer()] = m->initalizedBuffers[i];
//     }
//
//     for (std::size_t b = 0; b < m->dnx->buffers()->size(); ++b) {
//       if (mi->buffers[b].buffer != VK_NULL_HANDLE) {
//         // skip initalized buffers.
//         continue;
//       }
//       const dnx::Buffer *buffer = m->dnx->buffers()->Get(b);
//       std::size_t size =
//           evalUnsignedScalarSource(mi, buffer->size_type(), buffer->size());
//       // See if the buffer is a input / output buffer.
//       VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
//       for (std::size_t i = 0; i < m->dnx->inputs()->size(); ++i) {
//         const dnx::Input *input = m->dnx->inputs()->Get(i);
//         const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
//         if (tensor->buffer() == b) {
//           usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
//           break;
//         }
//       }
//       for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
//         const dnx::Output *output = m->dnx->outputs()->Get(o);
//         const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
//         if (tensor->buffer() == b) {
//           usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
//           break;
//         }
//       }
//       mi->buffers[b] = ctx->createBuffer(size, usage);
//       mi->ownedBuffers[b] = true;
//     }
//   }
//
//   *instance = mi;
//   return 0;
// }
//
// void destroy_runtime_model_instance(RuntimeContext context,
//                                     RuntimeModelInstance instance) {
//   auto ctx = reinterpret_cast<runtime::Context *>(context);
//   auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
//   for (std::size_t b = 0; b < mi->buffers.size(); ++b) {
//     if (mi->ownedBuffers[b]) {
//       ctx->destroyBuffer(mi->buffers[b]);
//     }
//   }
//   delete mi;
// }
//
// int eval_runtime_model_instance(RuntimeContext context,
//                                 RuntimeModelInstance instance, void **inputs,
//                                 void **outputs) {
//   auto ctx = reinterpret_cast<runtime::Context *>(context);
//   auto mi = reinterpret_cast<runtime::ModelInstance *>(instance);
//   auto m = mi->model;
//
//   // 1. Allocate output.
//   std::size_t outputCount = m->dnx->outputs()->size();
//   std::size_t inputCount = m->dnx->inputs()->size();
//
//   runtime::Buffer *outputStages = new runtime::Buffer[outputCount];
//   std::size_t *outputSizes = new std::size_t[outputCount];
//   for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
//     const dnx::Output *output = m->dnx->outputs()->Get(o);
//     const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
//     std::uint64_t size =
//         evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
//     outputSizes[o] = size;
//     outputStages[o] =
//         ctx->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
//   }
//
//   runtime::Buffer *inputStages = new runtime::Buffer[inputCount];
//   for (std::size_t i = 0; i < static_cast<std::size_t>(inputCount); ++i) {
//     const dnx::Input *input = m->dnx->inputs()->Get(i);
//     const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
//     std::uint64_t size =
//         evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
//     inputStages[i] = ctx->createBuffer(
//         size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
//     ctx->copy(inputStages[i].allocation, inputs[i], size);
//   }
//
//   VkCommandPool cmdPool = ctx->createCommandPool();
//   VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);
//
//   // upload input.
//   for (std::size_t i = 0; i < m->dnx->inputs()->size(); ++i) {
//     const dnx::Input *input = m->dnx->inputs()->Get(i);
//     const dnx::Tensor *tensor = m->dnx->tensors()->Get(input->tensor());
//     std::size_t size =
//         evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
//     std::size_t offset =
//         evalUnsignedScalarSource(mi, tensor->offset_type(), tensor->offset());
//     ctx->cmdCopy(cmd, mi->buffers[tensor->buffer()], inputStages[i], size,
//                  offset, 0);
//   }
//   // Wait for transfer to be done.
//   ctx->cmdMemoryBarrier(
//       cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
//       VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
//
//   // record all dispatches.
//   for (std::size_t d = 0; d < m->dnx->dispatches()->size(); ++d) {
//     switch (static_cast<dnx::Dispatch>(m->dnx->dispatches_type()->Get(d))) {
//     case dnx::Dispatch::Dispatch_NONE:
//       std::runtime_error("invalid state");
//       break;
//     case dnx::Dispatch_ComputeDispatch:
//       const dnx::ComputeDispatch *dispatch =
//           m->dnx->dispatches()->GetAs<dnx::ComputeDispatch>(d);
//       VkPipeline pipeline = m->pipelines[d];
//       VkPipelineLayout layout = m->pipelineLayouts[d];
//       assert(pipeline != VK_NULL_HANDLE);
//       assert(layout != VK_NULL_HANDLE);
//       vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
//
//       std::vector<std::byte> pushConstantBuffer =
//           parsePushConstant(mi, dispatch->push_constant());
//
//       vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
//                          pushConstantBuffer.size(), pushConstantBuffer.data());
//
//       // vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
//       //     layout, uint32_t firstSet,
//       //     uint32_t descriptorSetCount,
//       //     const VkDescriptorSet *pDescriptorSets,
//       //     uint32_t dynamicOffsetCount,
//       //     const uint32_t *pDynamicOffsets)
//
//       // vkCmdDispatch(cmd, 1, 1, 1);
//       break;
//     }
//   }
//
//   // Wait for compute to be done.
//   ctx->cmdMemoryBarrier(
//       cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
//       VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
//   // download output.
//   for (std::size_t o = 0; o < m->dnx->outputs()->size(); ++o) {
//     const dnx::Output *output = m->dnx->outputs()->Get(o);
//     const dnx::Tensor *tensor = m->dnx->tensors()->Get(output->tensor());
//     std::size_t size =
//         evalUnsignedScalarSource(mi, tensor->size_type(), tensor->size());
//     std::size_t offset =
//         evalUnsignedScalarSource(mi, tensor->offset_type(), tensor->offset());
//     ctx->cmdCopy(cmd, outputStages[o], mi->buffers[tensor->buffer()], size, 0,
//                  offset);
//   }
//   // Wait for transfer to be done.
//   ctx->cmdMemoryBarrier(
//       cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
//       VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
//
//   ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
//   ctx->destroyCommandPool(cmdPool);
//
//   for (std::size_t i = 0; i < inputCount; ++i) {
//     ctx->destroyBuffer(inputStages[i]);
//   }
//   delete[] inputStages;
//
//   for (std::size_t o = 0; o < outputCount; ++o) {
//     ctx->copy(outputs[o], outputStages[o].allocation, outputSizes[o]);
//     ctx->destroyBuffer(outputStages[o]);
//   }
//   delete[] outputStages;
//   delete[] outputSizes;
//
//   return 0;
// }

} // namespace denox

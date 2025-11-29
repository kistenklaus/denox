
#include "Db.hpp"
#include "context.hpp"
#include "denox/runtime.hpp"
#include "vma.hpp"
#include <alloca.h>
#include <chrono>
#include <fmt/base.h>
#include <forward_list>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <vulkan/vulkan_core.h>
using namespace std::chrono_literals;

namespace denox {

static constexpr std::chrono::duration SYNC_INTERVAL = 100ms;
static constexpr uint64_t SAMPLE_SIZE = 1;
static constexpr uint64_t PIPELINE_WARMUP_ITERATIONS = 50;
static constexpr size_t MEMORY_WARMUP_SIZE = 48000000; // 8MB
static constexpr bool WARMUP_CACHES = true;
static constexpr bool RANDOMIZED_INPUT = false;

static runtime::ComputeDispatch &select_target(runtime::Db &db) {
  uint64_t min_samples = std::numeric_limits<uint64_t>::max();
  size_t target_index = 0;

  for (size_t i = 0; i < db.dispatches.size(); ++i) {
    const auto &target = db.dispatches[i];
    if (target.time.samples < min_samples) {
      target_index = i;
      min_samples = target.time.samples;
    }
  }
  return db.dispatches[target_index];
}

static std::pair<VkPipelineLayout, std::map<uint32_t, VkDescriptorSetLayout>>
create_compute_dispatch_layouts(runtime::Context *ctx,
                                const runtime::ComputeDispatch &dispatch) {

  std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> bindings;
  for (const runtime::TensorBinding &binding : dispatch.bindings) {
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding.binding;
    layoutBinding.descriptorCount = 1;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[binding.set].push_back(layoutBinding);
  }

  std::map<uint32_t, VkDescriptorSetLayout> descriptorSetLayouts;
  std::vector<VkDescriptorSetLayout> layouts;
  for (const auto &[set, binding] : bindings) {
    VkDescriptorSetLayout layout = ctx->createDescriptorSetLayout(binding);
    layouts.push_back(layout);
    descriptorSetLayouts.emplace(set, layout);
  }

  VkPipelineLayout pipelineLayout =
      ctx->createPipelineLayout(layouts, dispatch.pushConstant.size());
  return std::make_pair(pipelineLayout, descriptorSetLayouts);
}
static VkPipeline create_compute_dispatch_pipeline(
    runtime::Context *ctx, const runtime::ComputeDispatch &dispatch,
    const runtime::Db &db, VkPipelineLayout pipelineLayout) {
  VkPipeline pipeline = ctx->createComputePipeline(
      pipelineLayout, db.binaries[dispatch.binaryId].source, "main");
  return pipeline;
}

static VkDescriptorPool
create_descriptor_pool(runtime::Context *ctx,
                       const runtime::ComputeDispatch &dispatch) {
  std::set<uint32_t> sets;
  for (const auto &binding : dispatch.bindings) {
    sets.insert(binding.set);
  }
  std::vector<VkDescriptorPoolSize> sizes;
  VkDescriptorPoolSize storageBufferDescriptors;
  storageBufferDescriptors.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  storageBufferDescriptors.descriptorCount = dispatch.bindings.size();
  sizes.push_back(storageBufferDescriptors);
  return ctx->createDescriptorPool(sets.size(), sizes);
}

static std::vector<runtime::Buffer>
allocate_buffers(runtime::Context *ctx,
                 const runtime::ComputeDispatch &dispatch) {
  std::vector<runtime::Buffer> buffers;
  buffers.reserve(dispatch.bindings.size());
  for (const runtime::TensorBinding &tensorBinding : dispatch.bindings) {
    if (tensorBinding.access == runtime::Access::ReadOnly ||
        tensorBinding.access == runtime::Access::WriteOnly) {
      runtime::Buffer buffer = ctx->createBuffer(
          tensorBinding.byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      buffers.push_back(buffer);
    } else {
      runtime::Buffer buffer = ctx->createBuffer(
          tensorBinding.byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      buffers.push_back(buffer);
    }
  }
  return buffers;
}

static std::vector<VkDescriptorSet> create_and_update_descriptor_sets(
    runtime::Context *ctx, VkDescriptorPool pool,
    const std::map<uint32_t, VkDescriptorSetLayout> &layouts,
    const runtime::ComputeDispatch &dispatch,
    std::span<const runtime::Buffer> buffers) {

  std::vector<uint32_t> setOrder;
  std::vector<VkDescriptorSetLayout> layoutsVec;
  for (const auto &[set, layout] : layouts) {
    layoutsVec.push_back(layout);
    setOrder.push_back(set);
  }
  std::vector<VkDescriptorSet> setVec(layouts.size());
  ctx->allocDescriptorSets(pool, layoutsVec, setVec.data());
  std::map<uint32_t, VkDescriptorSet> sets;
  for (size_t i = 0; i < setOrder.size(); ++i) {
    sets.emplace(setOrder[i], setVec[i]);
  }

  std::forward_list<VkDescriptorBufferInfo> bufferInfosArena;

  std::vector<VkWriteDescriptorSet> writeInfos;
  for (size_t i = 0; i < dispatch.bindings.size(); ++i) {
    const auto &tensorBinding = dispatch.bindings[i];
    VkWriteDescriptorSet writeInfo{};
    writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeInfo.pNext = nullptr;
    writeInfo.dstSet = sets[tensorBinding.set];
    writeInfo.dstBinding = tensorBinding.binding;
    writeInfo.dstArrayElement = 0;
    writeInfo.descriptorCount = 1;
    writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    auto &bufferInfo = bufferInfosArena.emplace_front();
    bufferInfo.buffer = buffers[i].vkbuffer;
    bufferInfo.range = tensorBinding.byteSize;
    bufferInfo.offset = 0;
    writeInfo.pBufferInfo = &bufferInfo;

    writeInfos.push_back(writeInfo);
  }

  ctx->updateDescriptorSets(writeInfos);
  return setVec;
}

static void print_state(const runtime::Db &db) {
  uint64_t minSampleCount = std::numeric_limits<uint64_t>::max();
  uint64_t totalSampleCount = 0;
  uint64_t maxStdDerivation = 0;
  for (const auto &target : db.dispatches) {
    totalSampleCount += target.time.samples;
    if (target.time.samples < minSampleCount) {
      minSampleCount = target.time.samples;
    }
    if (target.time.samples > 0 &&
        target.time.std_derivation_ns > maxStdDerivation) {
      maxStdDerivation = target.time.std_derivation_ns;
    }
  }
  fmt::println("Min-Sample-Count : {}", minSampleCount);
  fmt::println("Mean-Sample-Count: {:.3f}",
               static_cast<float>(totalSampleCount) / db.dispatches.size());
  fmt::println("Max-StdDerivation: ±{:.3f}ms", maxStdDerivation / 1e6f);
}

int bench_runtime_instance(RuntimeContext context, const char *dbfile,
                           size_t minSamples) {

  if (minSamples == 0) {
    minSamples = std::numeric_limits<uint32_t>::max();
  }

  auto *ctx = reinterpret_cast<denox::runtime::Context *>(context);

  if (dbfile == nullptr) {
    return -1;
  }

  std::string dbpath{dbfile};
  auto db = runtime::Db::open(dbpath);
  if (db.dispatches.empty()) {
    return 0;
  }

  auto last_sync = std::chrono::high_resolution_clock::now();

  auto cmdPool = ctx->createCommandPool();

  runtime::Buffer warmup0, warmup1;
  if (MEMORY_WARMUP_SIZE != 0) {
    warmup0 = ctx->createBuffer(MEMORY_WARMUP_SIZE,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    warmup1 = ctx->createBuffer(MEMORY_WARMUP_SIZE,
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
  }

  size_t maxBufferSize = 0;
  for (const auto &target : db.dispatches) {
    for (const auto &buffer : target.bindings) {
      maxBufferSize = std::max(maxBufferSize, buffer.byteSize);
    }
  }
  runtime::Buffer warmupInputs;
  if (WARMUP_CACHES) {
    warmupInputs = ctx->createBuffer(
        maxBufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    if (RANDOMIZED_INPUT) {
      runtime::Buffer stage = ctx->createBuffer(
          maxBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
      std::vector<uint8_t> bytes(maxBufferSize);
      std::random_device rng;
      std::mt19937 prng{rng()};
      std::uniform_int_distribution<uint8_t> dist;
      for (auto &b : bytes) {
        b = dist(prng);
      }

      ctx->copy(stage.allocation, bytes.data(), bytes.size());

      VkCommandBuffer cmd = ctx->allocBeginCommandBuffer(cmdPool);
      ctx->cmdCopy(cmd, warmupInputs, stage, maxBufferSize);
      ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);
      ctx->destroyBuffer(stage);
    }
  }
  print_state(db);

  while (true) {
    // Select benchmark target
    runtime::ComputeDispatch &target = select_target(db);
    // runtime::ComputeDispatch &target = db.dispatches[2];

    if (target.time.samples >= minSamples) {
      break;
    }

    // Initalize vulkan resources.
    auto [pipelineLayout, descriptorSetLayouts] =
        create_compute_dispatch_layouts(ctx, target);

    auto pipeline =
        create_compute_dispatch_pipeline(ctx, target, db, pipelineLayout);

    auto descriptorPool = create_descriptor_pool(ctx, target);

    auto buffers = allocate_buffers(ctx, target);

    auto descriptorSets = create_and_update_descriptor_sets(
        ctx, descriptorPool, descriptorSetLayouts, target, buffers);

    uint32_t queryCount = 2;
    VkQueryPool queryPool = ctx->createTimestampQueryPool(queryCount);

    // Record GPU execution.
    auto cmd = ctx->allocBeginCommandBuffer(cmdPool);
    ctx->cmdResetQueryPool(cmd, queryPool, 0, queryCount);

    if (MEMORY_WARMUP_SIZE > 0) { // warmup memory. (big memcpy, to warm memory
                                  // clock and flush caches)
      ctx->cmdCopy(cmd, warmup0, warmup1, MEMORY_WARMUP_SIZE);
      ctx->cmdMemoryBarrier(
          cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      std::swap(warmup0, warmup1);
    }

    if (PIPELINE_WARMUP_ITERATIONS > 0) { // compute warmup.
      for (size_t it = 0; it < PIPELINE_WARMUP_ITERATIONS; ++it) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           target.pushConstant.size(),
                           target.pushConstant.data());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, descriptorSets.size(),
                                descriptorSets.data(), 0, nullptr);
        vkCmdDispatch(cmd, target.workgroupCountX, target.workgroupCountY,
                      target.workgroupCountZ);
      }
      ctx->cmdMemoryBarrier(
          cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    if (WARMUP_CACHES) { // warmup caches
      for (size_t b = 0; b < target.bindings.size(); ++b) {
        const auto &tensorBinding = target.bindings[b];
        if (tensorBinding.access == runtime::Access::ReadOnly ||
            tensorBinding.access == runtime::Access::ReadWrite) {
          ctx->cmdCopy(cmd, buffers[b], warmupInputs, tensorBinding.byteSize);
        }
      }
    }
    if (WARMUP_CACHES || MEMORY_WARMUP_SIZE > 0) {

      ctx->cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_TRANSFER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT);
    }

    // ctx->cmdMemoryBarrier(
    //     cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
    //     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                           0);
    for (size_t it = 0; it < SAMPLE_SIZE; ++it) {

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         target.pushConstant.size(),
                         target.pushConstant.data());

      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipelineLayout, 0, descriptorSets.size(),
                              descriptorSets.data(), 0, nullptr);
      vkCmdDispatch(cmd, target.workgroupCountX, target.workgroupCountY,
                    target.workgroupCountZ);
    }
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                           1);

    ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);

    // Interpret results.
    std::vector<uint64_t> timestamps = ctx->getQueryResults(queryPool, 2);
    if (timestamps[1] > timestamps[0]) {

      uint64_t totalLatency = ctx->timestampNanoDifference(timestamps, 0, 1);

      float duration = ctx->timestampDifference(timestamps, 0, 1);
      if (totalLatency > 1e11) {
        fmt::println("Timestamps wrapped trying again");
        continue; // just retry.
      }
      double mean_new = double(totalLatency) / double(SAMPLE_SIZE);
      constexpr double var_new = 0.0;
      constexpr double std_new = 0.0;
      constexpr double m = 1.0; // new samples.

      uint64_t old_samples = target.time.samples;

      if (old_samples > 0) {
        double n = double(old_samples);

        double mean_old = double(target.time.latency_ns);
        double std_old = double(target.time.std_derivation_ns);
        double var_old = std_old * std_old;

        double mean_total = (n * mean_old + m * mean_new) / (n + m);

        double var_total =
            (n * (var_old + (mean_old - mean_total) * (mean_old - mean_total)) +
             m * (var_new +
                  (mean_new - mean_total) * (mean_new - mean_total))) /
            (n + m);

        target.time.samples = uint64_t(n + m);

        target.time.latency_ns = uint64_t(mean_total + 0.5);
        target.time.std_derivation_ns = uint64_t(std::sqrt(var_total) + 0.5);
        // fmt::println("std_derivation  {}ms",
        //              target.time.std_derivation_ns / 1e6);
        //
        // fmt::println("latency  {}ms", target.time.latency_ns / 1e6);

      } else {
        target.time.samples = 1;
        target.time.latency_ns = uint64_t(mean_new + 0.5);
        target.time.std_derivation_ns = 0;
      }
    } else {
      fmt::println("WARNING: Wrapping timestamps. trying again");
    }

    { // cleanup
      ctx->destroyQueryPool(queryPool);

      for (const auto &buffer : buffers) {
        ctx->destroyBuffer(buffer);
      }

      ctx->destroyDescriptorPool(descriptorPool);
      ctx->destroyPipeline(pipeline);

      ctx->destroyPipelineLayout(pipelineLayout);
      for (const auto &[_, layout] : descriptorSetLayouts) {
        ctx->destroyDescriptorSetLayout(layout);
      }
    }

    // fmt::println("");

    auto time_since_last_sync =
        std::chrono::high_resolution_clock::now() - last_sync;
    if (time_since_last_sync > SYNC_INTERVAL) {
      db.write_back();
      print_state(db);
      last_sync = std::chrono::high_resolution_clock::now();
    }
  }
  print_state(db);

  if (WARMUP_CACHES) {
    ctx->destroyBuffer(warmupInputs);
  }
  if (MEMORY_WARMUP_SIZE != 0) {
    ctx->destroyBuffer(warmup0);
    ctx->destroyBuffer(warmup1);
  }
  ctx->destroyCommandPool(cmdPool);

  db.write_back();

  return 0;
}

} // namespace denox

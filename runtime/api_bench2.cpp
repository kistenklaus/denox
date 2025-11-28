
#include "Db.hpp"
#include "context.hpp"
#include "denox/runtime.hpp"
#include <algorithm>
#include <alloca.h>
#include <chrono>
#include <cmath>
#include <fmt/base.h>
#include <forward_list>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vulkan/vulkan_core.h>
using namespace std::chrono_literals;

namespace denox {

static constexpr std::chrono::duration SYNC_INTERVAL = 5s;
static constexpr size_t SAMPLE_BATCH = 100;

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
    runtime::Buffer buffer = ctx->createBuffer(
        tensorBinding.byteSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buffers.push_back(buffer);
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

/// Rounded interger square root.
static uint64_t irsqrt(uint64_t x) {
  uint64_t left = 0;
  uint64_t right = uint64_t(1) << 32;

  while (left < right) {
    uint64_t mid = (left + right + 1) >> 1;
    if ((unsigned __int128)mid * mid <= x)
      left = mid;
    else
      right = mid - 1;
  }

  uint64_t r = left;
  uint64_t r2 = r * r;
  uint64_t rp = r + 1;
  uint64_t rp2 = (rp > r) ? rp * rp : UINT64_MAX;
  uint64_t err_r = x - r2;
  uint64_t err_rp = (rp2 > x) ? (rp2 - x) : (x - rp2);

  if (err_rp < err_r)
    return rp;
  else
    return r;
}

int bench_runtime_instance(RuntimeContext context, const char *dbfile,
                           size_t minSamples) {
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

  while (true) {
    // Select benchmark target
    runtime::ComputeDispatch &target = select_target(db);
    fmt::println("Benchmarking Shader-ID: {}", target.binaryId);


    // Initalize vulkan resources.
    auto [pipelineLayout, descriptorSetLayouts] =
        create_compute_dispatch_layouts(ctx, target);

    auto pipeline =
        create_compute_dispatch_pipeline(ctx, target, db, pipelineLayout);

    auto descriptorPool = create_descriptor_pool(ctx, target);

    auto buffers = allocate_buffers(ctx, target);

    auto descriptorSets = create_and_update_descriptor_sets(
        ctx, descriptorPool, descriptorSetLayouts, target, buffers);

    uint32_t samples = SAMPLE_BATCH;

    uint32_t queryCount = samples + 1;
    VkQueryPool queryPool = ctx->createTimestampQueryPool(queryCount);

    // Record GPU execution.
    auto cmd = ctx->allocBeginCommandBuffer(cmdPool);
    ctx->cmdResetQueryPool(cmd, queryPool, 0, queryCount);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       target.pushConstant.size(), target.pushConstant.data());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                            0, descriptorSets.size(), descriptorSets.data(), 0,
                            nullptr);

    uint32_t query = 0;
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                           query++);
    for (size_t it = 0; it < samples; ++it) {
      vkCmdDispatch(cmd, target.workgroupCountX, target.workgroupCountY,
                    target.workgroupCountZ);
      ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, queryPool,
                             query++);
    }

    ctx->endSubmitWaitCommandBuffer(cmdPool, cmd);

    // Interpret results.
    std::vector<uint64_t> timestamps =
        ctx->getQueryResults(queryPool, queryCount);

    std::vector<uint64_t> latencies(samples);
    for (size_t i = 0; i < samples; ++i) {
      latencies[i] = ctx->timestampNanoDifference(timestamps, i, i + 1);
    }
    uint64_t total_latency = 0;
    for (const auto &sample : latencies) {
      total_latency += sample;
    }
    uint64_t mean_latency = total_latency / samples;
    uint64_t variance_acc = 0;
    for (const auto &sample : latencies) {
      int64_t diff =
          static_cast<int64_t>(sample) - static_cast<int64_t>(mean_latency);
      variance_acc += diff * diff;
    }
    uint64_t variance = variance_acc / samples;
    uint64_t std_derivation = irsqrt(variance);

    uint64_t old_sample_count = target.time.samples;
    uint64_t old_latency = target.time.latency_ns;
    uint64_t old_std_derivation = target.time.std_derivation_ns;

    uint64_t new_sample_count = old_sample_count + samples;
    uint64_t new_mean_latency =
        (old_sample_count * old_latency + total_latency) / new_sample_count;

    uint64_t new_std_derivation;
    {
      int64_t n = static_cast<int64_t>(old_sample_count);
      int64_t m = static_cast<int64_t>(samples);
      int64_t uo = static_cast<int64_t>(old_latency);
      int64_t un = static_cast<int64_t>(mean_latency);
      int64_t u = static_cast<int64_t>(new_mean_latency);
      int64_t so = static_cast<int64_t>(old_std_derivation);
      int64_t sn = static_cast<int64_t>(std_derivation);
      int64_t vo = so * so;
      int64_t vn = sn * sn;

      int64_t r0 = uo - u;
      int64_t r1 = r0 * r0;
      int64_t r2 = vo + r1;
      int64_t r3 = n * r2;

      int64_t r4 = un - u;
      int64_t r5 = r4 * r4;
      int64_t r6 = vn * vn;
      int64_t r7 = m * r6;

      int64_t num = r7 + r3;
      int64_t denom = n + m;

      // rounding integer division.
      int64_t v = (num + denom / 2) / denom;
      new_std_derivation = irsqrt(v);
    }
    target.time.samples = new_sample_count;
    target.time.latency_ns = new_mean_latency;
    target.time.std_derivation_ns = new_std_derivation;

    fmt::println("  samples: {}", new_sample_count);
    fmt::println("  latency: {:.3f}ms", new_mean_latency / 1e6f);

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

    auto time_since_last_sync =
        std::chrono::high_resolution_clock::now() - last_sync;
    if (time_since_last_sync > SYNC_INTERVAL) {
      db.write_back();
      fmt::println("Writing results back");
      last_sync = std::chrono::high_resolution_clock::now();
    }
  }

  db.write_back();

  ctx->destroyCommandPool(cmdPool);
  return 0;
}

} // namespace denox

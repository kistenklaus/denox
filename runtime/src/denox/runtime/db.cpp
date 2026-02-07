#include "denox/runtime/db.hpp"
#include "denox/algorithm/align_up.hpp"
#include "denox/common/commit_hash.hpp"
#include "denox/common/os.hpp"
#include "denox/common/version.hpp"
#include "denox/db/DbComputeDispatch.hpp"
#include "denox/db/DbEnv.hpp"
#include "denox/device_info/query/query_driver_device_info.hpp"
#include "denox/diag/logging.hpp"
#include "denox/glsl/GlslCompiler.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/runtime/context.hpp"
#include "denox/spirv/SpirvBinary.hpp"
#include "denox/spirv/SpirvTools.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fmt/ostream.h>
#include <forward_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <semaphore>
#include <thread>
#include <vulkan/vulkan_core.h>

// 100, 1000, 100 with jit=2 worked well, but didn't converge completely.

// static constexpr size_t EPOCH_SAMPLES = EPOCH_SIZE * 10;

static constexpr size_t ASYNC_EPOCH_DEPTH = 3;

static constexpr size_t JIT_WARMUP_ITERATIONS = 0;
static constexpr size_t L2_WARMUP_ITERATONS = 0;
static constexpr size_t PIPELINE_STAGES = 2;

using namespace denox;

struct BenchmarkState {
  std::mt19937 prng;
  std::unique_ptr<spirv::SpirvTools> tools;
  std::unique_ptr<spirv::GlslCompiler> glslCompiler;
};

static uint64_t benchmark_timestamp() {
  return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::nano>>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static BenchmarkState
create_benchmark_state(const runtime::ContextHandle &ctx) {
  std::random_device rng;
  std::mt19937 prng(rng());

  auto device_info = denox::query_driver_device_info(
      ctx->vkInstance(), ctx->vkPhysicalDevice(), ApiVersion::VULKAN_1_4);
  auto tools = std::make_unique<spirv::SpirvTools>(device_info);
  auto glslCompiler =
      std::make_unique<spirv::GlslCompiler>(tools.get(), device_info);

  return BenchmarkState{
      .prng = std::move(prng),
      .tools = std::move(tools),
      .glslCompiler = std::move(glslCompiler),
  };
}

static memory::vector<uint32_t> select_targets_from_candidates(
    BenchmarkState &state, const denox::Db &db,
    memory::span<const uint32_t> candidates, size_t N, uint32_t minSamples,
    float maxRelativeError, bool selectUnique = true,
    memory::optional<memory::span<const uint64_t>> samplesInFlight =
        memory::nullopt) {
  memory::vector<uint32_t> result;
  result.reserve(N);

  const auto dispatches = db.dispatches();

  auto inflight = [&](uint32_t candidateIndex) -> uint64_t {
    if (!samplesInFlight.has_value())
      return 0;
    return (*samplesInFlight)[candidateIndex];
  };

  // ------------------------------------------------------------
  // Phase 1: cold-first (minSamples), accounting for in-flight
  // ------------------------------------------------------------
  for (uint32_t i = 0; i < candidates.size() && result.size() < N; ++i) {
    uint32_t d = candidates[i]; // DB dispatch index

    uint64_t dbSamples = 0;
    if (dispatches[d].time.has_value())
      dbSamples = dispatches[d].time->samples.size();

    uint64_t effectiveSamples = dbSamples + inflight(i);

    if (effectiveSamples >= minSamples) {
      continue;
    }
    if (selectUnique) {
      result.push_back(i);
      continue;
    }

    uint64_t required = minSamples - effectiveSamples;
    uint64_t remaining = N - result.size();
    uint64_t count = std::min(required, remaining);

    for (uint32_t k = 0; k < count; ++k)
      result.push_back(i); // index into candidates
  }

  if (result.size() == N) {
    std::ranges::shuffle(result, state.prng);
    return result;
  }
  // ------------------------------------------------------------
  // Phase 2: SEM-based selection (no replacement, no in-flight,
  // exclude already-converged dispatches)
  // ------------------------------------------------------------
  struct Item {
    uint32_t candidateIndex; // index into candidates
    double priority;
  };

  const uint64_t remaining = N - result.size();

  memory::vector<Item> items;
  items.reserve(candidates.size());

  for (uint32_t i = 0; i < candidates.size(); ++i) {
    uint32_t d = candidates[i];
    const auto &dispatch = dispatches[d];

    // Exclude dispatches with no data yet from SEM phase
    if (!dispatch.time.has_value()) {
      items.emplace_back(i, std::numeric_limits<double>::infinity());
      continue;
    }

    const auto &t = *dispatch.time;
    const uint64_t n = t.samples.size();

    if (n > 1 && t.std_derivation_ns > 0 && t.mean_latency_ns > 0) {
      const double sem = static_cast<double>(t.std_derivation_ns) /
                         std::sqrt(static_cast<double>(n));
      const double relError = sem / static_cast<double>(t.mean_latency_ns);

      const bool precisionOk =
          (relError <= static_cast<double>(maxRelativeError));
      if (!precisionOk) {
        const double priority = sem / static_cast<double>(t.mean_latency_ns);
        items.emplace_back(i, priority);
      }
    } else {
      items.emplace_back(i, std::numeric_limits<double>::infinity());
    }
  }

  const uint64_t take = std::min<uint64_t>(items.size(), remaining);

  std::partial_sort(
      items.begin(), items.begin() + static_cast<std::ptrdiff_t>(take),
      items.end(),
      [](const Item &a, const Item &b) { return a.priority > b.priority; });

  for (uint32_t k = 0; k < take; ++k)
    result.push_back(items[k].candidateIndex);

  std::ranges::shuffle(result, state.prng);
  return result;
}

struct EpochDispatch {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  memory::vector<VkDescriptorSetLayout> descriptorLayouts;
  memory::vector<VkDescriptorSet> descriptorSets; // ordered by set index.
  memory::span<const uint8_t> pc;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
};

struct EpochStage {
  VkCommandBuffer cmd;
  VkQueryPool queryPool;
  VkFence fence;
  uint32_t env;
};

struct Epoch {
  uint32_t batchSize;
  uint32_t sample_count;
  memory::vector<uint32_t> targets;
  VkCommandPool cmdPool;
  runtime::Buffer buffer;
  memory::vector<VkPipeline> pipelines;
  memory::vector<VkPipelineLayout> pipelineLayouts;
  memory::vector<VkDescriptorSetLayout> descriptorLayouts;
  VkDescriptorPool descriptorPool;
  memory::vector<EpochDispatch> dispatches;
  std::array<EpochStage, PIPELINE_STAGES> stages;
  uint32_t env;
};

struct Timing {
  uint32_t target;
  std::vector<DbSample> samples;
};

struct EpochBenchResults {
  memory::vector<Timing> timings;
};

static Epoch create_epoch(const runtime::ContextHandle &ctx,
                          const denox::Db &db, memory::span<uint32_t> targets,
                          uint32_t env, uint32_t batchSize,
                          uint32_t sampleCount) {

  const auto dbdispatches = db.dispatches();
  const auto dbbinaries = db.binaries();

  static size_t jj = std::thread::hardware_concurrency();
  // static size_t jj = 1;

  memory::vector<uint32_t> localMaxSets(jj);
  memory::vector<uint32_t> localStorageBufferDescriptorCount(jj);
  memory::vector<size_t> localPeakBufferSize(jj);

  memory::vector<EpochDispatch> dispatches(targets.size(), EpochDispatch{});

  // auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads(jj);
  for (size_t tid = 0; tid < threads.size(); ++tid) {
    threads[tid] = std::thread([&, tid]() {
      localMaxSets[tid] = 0;
      localStorageBufferDescriptorCount[tid] = 0;
      localPeakBufferSize[tid] = 0;
      size_t start = (tid * targets.size()) / threads.size();
      size_t end = ((tid + 1) * targets.size()) / threads.size();
      if (tid == threads.size() - 1) {
        end = targets.size();
      }

      for (size_t i = start; i < end; ++i) {
        uint32_t target = targets[i];
        const auto &dbdispatch = dbdispatches[target];

        size_t totalBufferSize = 0;
        uint32_t maxSet = 0;
        for (const auto &binding : dbdispatch.bindings) {
          totalBufferSize =
              algorithm::align_up(totalBufferSize, binding.alignment);
          totalBufferSize += binding.byteSize;
          maxSet = std::max(maxSet, binding.set);
          localStorageBufferDescriptorCount[tid] += 1;
        }
        const uint32_t setCount = maxSet + 1;
        localMaxSets[tid] += setCount;
        localPeakBufferSize[tid] =
            std::max(localPeakBufferSize[tid], totalBufferSize);

        const uint32_t binaryId = dbdispatch.binaryId;

        memory::small_vector<
            memory::small_vector<VkDescriptorSetLayoutBinding, 4>, 4>
            bindings(setCount);
        for (const auto &binding : dbdispatch.bindings) {
          bindings[binding.set].push_back(VkDescriptorSetLayoutBinding{
              .binding = binding.binding,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .descriptorCount = 1,
              .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
              .pImmutableSamplers = nullptr,
          });
        }

        memory::vector<VkDescriptorSetLayout> setLayouts(setCount);
        for (uint32_t set = 0; set < setCount; ++set) {
          setLayouts[set] = ctx->createDescriptorSetLayout(bindings[set]);
        }

        VkPipelineLayout layout = ctx->createPipelineLayout(
            setLayouts, static_cast<uint32_t>(dbdispatch.pushConstant.size()));

        VkPipeline pipeline = ctx->createComputePipeline(
            layout, dbbinaries[binaryId].spvBinary.spv, "main");

        dispatches[i] = EpochDispatch{
            .pipeline = pipeline,
            .layout = layout,
            .descriptorLayouts = setLayouts,
            .descriptorSets = {}, // <- allocated later
            .pc = dbdispatch.pushConstant,
            .workgroupCountX = dbdispatch.workgroupCountX,
            .workgroupCountY = dbdispatch.workgroupCountY,
            .workgroupCountZ = dbdispatch.workgroupCountZ,
        };
      }
    });
  }
  for (uint32_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }

  // auto end = std::chrono::high_resolution_clock::now();
  // auto dur =
  //     std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
  //         end - start);
  // fmt::println("took {}", dur);

  memory::vector<VkPipeline> pipelines;
  memory::vector<VkPipelineLayout> pipelineLayouts;
  memory::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  for (const auto &dispatch : dispatches) {
    pipelines.push_back(dispatch.pipeline);
    pipelineLayouts.push_back(dispatch.layout);
    for (const auto &dlayout : dispatch.descriptorLayouts) {
      descriptorSetLayouts.push_back(dlayout);
    }
  }

  size_t peakBufferSize = 0;
  uint32_t storageBufferDescriptorCount = 0;
  uint32_t maxSets = 0;
  for (uint32_t tid = 0; tid < threads.size(); ++tid) {
    peakBufferSize = std::max(peakBufferSize, localPeakBufferSize[tid]);
    storageBufferDescriptorCount += localStorageBufferDescriptorCount[tid];
    maxSets += localMaxSets[tid];
  }

  runtime::Buffer buffer =
      ctx->createBuffer(peakBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  memory::vector<VkDescriptorPoolSize> descriptorPoolSizes{
      VkDescriptorPoolSize{
          .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = storageBufferDescriptorCount,
      },
  };

  VkDescriptorPool descriptorPool =
      ctx->createDescriptorPool(maxSets, descriptorPoolSizes);

  for (uint32_t x = 0; x < targets.size(); ++x) {
    const uint32_t target = targets[x];
    const auto &dbdispatch = dbdispatches[target];
    auto &dispatch = dispatches[x];
    uint32_t setCount =
        static_cast<uint32_t>(dispatch.descriptorLayouts.size());
    dispatch.descriptorSets.resize(setCount);
    ctx->allocDescriptorSets(descriptorPool, dispatch.descriptorLayouts,
                             dispatch.descriptorSets.data());

    std::forward_list<VkDescriptorBufferInfo> monotone_allocator;
    memory::vector<VkWriteDescriptorSet> writeInfos;
    writeInfos.reserve(dbdispatch.bindings.size());
    uint64_t offset = 0;
    for (const auto &binding : dbdispatch.bindings) {
      auto &writeInfo = writeInfos.emplace_back();
      writeInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writeInfo.pNext = nullptr;
      writeInfo.dstSet = dispatch.descriptorSets[binding.set];
      writeInfo.dstBinding = binding.binding;
      writeInfo.dstArrayElement = 0;
      writeInfo.descriptorCount = 1;
      writeInfo.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writeInfo.pImageInfo = nullptr;
      offset = algorithm::align_up(offset, binding.alignment);
      auto &bufferInfo = monotone_allocator.emplace_front();
      bufferInfo.buffer = buffer.vkbuffer;
      bufferInfo.range = binding.byteSize;
      bufferInfo.offset = offset;
      offset += binding.byteSize;
      writeInfo.pBufferInfo = &bufferInfo;
      ;
      writeInfo.pTexelBufferView = nullptr;
    }
    ctx->updateDescriptorSets(writeInfos);
  }

  VkCommandPool cmdPool = ctx->createCommandPool();

  std::array<EpochStage, PIPELINE_STAGES> stages;
  for (size_t i = 0; i < stages.size(); ++i) {
    stages[i].cmd = ctx->allocCommandBuffer(cmdPool);
    stages[i].queryPool = ctx->createTimestampQueryPool(batchSize * 2);
    stages[i].fence = ctx->createFence(true);
    stages[i].env = env;
  }

  return Epoch{
      .batchSize = batchSize,
      .sample_count = sampleCount,
      .targets = {targets.begin(), targets.end()},
      .cmdPool = cmdPool,
      .buffer = buffer,
      .pipelines = std::move(pipelines),
      .pipelineLayouts = std::move(pipelineLayouts),
      .descriptorLayouts = std::move(descriptorSetLayouts),
      .descriptorPool = descriptorPool,
      .dispatches = std::move(dispatches),
      .stages = std::move(stages),
      .env = env,
  };
}

static void destroy_epoch(const runtime::ContextHandle &ctx, Epoch epoch) {
  for (const auto pipeline : epoch.pipelines) {
    ctx->destroyPipeline(pipeline);
  }
  for (const auto pipelineLayout : epoch.pipelineLayouts) {
    ctx->destroyPipelineLayout(pipelineLayout);
  }
  for (const auto setLayout : epoch.descriptorLayouts) {
    ctx->destroyDescriptorSetLayout(setLayout);
  }
  ctx->destroyCommandPool(epoch.cmdPool);
  ctx->destroyBuffer(epoch.buffer);

  ctx->destroyDescriptorPool(epoch.descriptorPool);
  for (const auto &stage : epoch.stages) {
    ctx->destroyQueryPool(stage.queryPool);
    ctx->destroyFence(stage.fence);
  }
}

struct Batch {
  bool live;
  memory::vector<uint32_t> dispatches; // <- indexes into EpochDispatch
};

static Batch create_batch(BenchmarkState &state, const denox::Db &db,
                          const Epoch &epoch, uint32_t minSamples,
                          float maxRelativeError,
                          memory::span<uint64_t> samplesInFlight) {
  memory::vector<uint32_t> dispatches = select_targets_from_candidates(
      state, db, epoch.targets, epoch.batchSize, minSamples, maxRelativeError,
      false, samplesInFlight);
  for (uint32_t x : dispatches) {
    samplesInFlight[x] += 1;
  }

  return Batch{
      .live = false,
      .dispatches = std::move(dispatches),
  };
}

static void record_dispatch(VkCommandBuffer cmd,
                            const EpochDispatch &dispatch) {
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dispatch.layout,
                          0,
                          static_cast<uint32_t>(dispatch.descriptorSets.size()),
                          dispatch.descriptorSets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, dispatch.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     static_cast<uint32_t>(dispatch.pc.size()),
                     dispatch.pc.data());
  vkCmdDispatch(cmd, dispatch.workgroupCountX, dispatch.workgroupCountY,
                dispatch.workgroupCountZ);
}

static void record_batch(VkCommandBuffer cmd, const runtime::ContextHandle &ctx,
                         const Epoch &epoch, const EpochStage &stage,
                         const Batch &batch) {

  // jit warmup
  for (uint32_t d : batch.dispatches) {
    const auto &dispatch = epoch.dispatches[d];
    for (size_t i = 0; i < JIT_WARMUP_ITERATIONS; ++i) {
      record_dispatch(cmd, dispatch);
    }
  }

  for (uint32_t t = 0; t < batch.dispatches.size(); ++t) {
    uint32_t d = batch.dispatches[t];
    const auto &dispatch = epoch.dispatches[d];
    // l2 warmup
    for (size_t i = 0; i < L2_WARMUP_ITERATONS; ++i) {
      record_dispatch(cmd, dispatch);
    }
    // ctx->cmdMemoryBarrierComputeShader(cmd);
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           stage.queryPool, t * 2);
    record_dispatch(cmd, dispatch);
    ctx->cmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                           stage.queryPool, t * 2 + 1);
    // ctx->cmdMemoryBarrierComputeShader(cmd);
  }
}

static void read_batch(const runtime::ContextHandle &ctx,
                       const EpochStage &stage, const Batch &batch,
                       memory::span<Timing> timings) {
  const size_t N = batch.dispatches.size();
  const uint32_t QUERY_COUNT = static_cast<uint32_t>(N * 2);

  memory::vector<uint64_t> timestamps =
      ctx->getQueryResults(stage.queryPool, QUERY_COUNT);

  for (size_t i = 0; i < N; ++i) {
    const uint32_t epochDispatchIndex = batch.dispatches[i];
    Timing &timing = timings[epochDispatchIndex];

    const uint32_t t0 = static_cast<uint32_t>(i * 2 + 0);
    const uint32_t t1 = static_cast<uint32_t>(i * 2 + 1);

    uint64_t ts0 = timestamps[t0];
    uint64_t ts1 = timestamps[t1];

    // Defensive checks (same semantics as old code)
    if (ts1 <= ts0) {
      // Timestamp wrap or invalid ordering
      continue;
    }

    uint64_t latency_ns = ctx->timestampNanoDifference(timestamps, t0, t1);

    // fmt::println("latency: {}ms", static_cast<float>(latency_ns) * 1e-6f);

    // ~100 ms sanity guard (same as before)
    if (latency_ns > 100'000'000'000ull) {
      continue;
    }

    // Incremental mean / variance update (Welford-style, batch weight = 1)
    std::chrono::high_resolution_clock::now().time_since_epoch();
    timing.samples.push_back(DbSample{
        .timestamp = benchmark_timestamp(),
        .latency_ns = latency_ns,
        .env = stage.env,
    });
  }
}

static EpochBenchResults bench_epoch(BenchmarkState &state, const denox::Db &db,
                                     const runtime::ContextHandle &ctx,
                                     const Epoch &epoch,
                                     const runtime::DbBenchOptions &options,
                                     uint32_t samples) {
  memory::vector<Timing> timings(epoch.targets.size());
  for (uint32_t i = 0; i < timings.size(); ++i) {
    timings[i].target = epoch.targets[i];
    timings[i].samples.clear();
  }
  std::array<Batch, PIPELINE_STAGES> batches;
  size_t stage = PIPELINE_STAGES - 1;

  uint64_t sampleCount = 0;
  memory::vector<uint64_t> samplesInFlight(epoch.targets.size(), 0);

  while (sampleCount < samples) {
    size_t next = (stage + 1) % PIPELINE_STAGES;
    ctx->waitFence(epoch.stages[next].fence);
    batches[next].live = false;

    ctx->resetFence(epoch.stages[next].fence);
    if (!batches[next].dispatches.empty()) {
      read_batch(ctx, epoch.stages[next], batches[next], timings);
    }

    batches[next] = create_batch(state, db, epoch, options.minSamples,
                                 options.maxRelativeError, samplesInFlight);
    sampleCount += batches[next].dispatches.size();
    if (batches[next].dispatches.empty()) {
      break;
    }

    assert(!batches[next].dispatches.empty());

    VkCommandBuffer cmd = epoch.stages[next].cmd;
    ctx->resetCommandBuffer(cmd);
    ctx->beginCommandBuffer(cmd);
    ctx->cmdResetQueryPool(cmd, epoch.stages[next].queryPool, 0,
                           epoch.batchSize * 2);
    record_batch(epoch.stages[next].cmd, ctx, epoch, epoch.stages[next],
                 batches[next]);
    ctx->endCommandBuffer(epoch.stages[next].cmd);
    // fmt::println("submitting: {}", next);
    batches[next].live = true;
    ctx->submit(epoch.stages[next].cmd, epoch.stages[next].fence);
    stage = next;
  }

  for (size_t i = 0; i < PIPELINE_STAGES; ++i) {
    if (batches[i].live) {
      // fmt::println("final-wait: {}", i);
      ctx->waitFence(epoch.stages[i].fence);
      if (!batches[i].dispatches.empty()) {
        read_batch(ctx, epoch.stages[i], batches[i], timings);
      }
    }
  }
  return EpochBenchResults{
      .timings = std::move(timings),
  };
}

static bool print_progress_report(const denox::Db &db,
                                  const runtime::DbBenchOptions &options,
                                  diag::Logger &logger) {
  const auto dispatches = db.dispatches();
  const uint64_t total = dispatches.size();

  uint64_t noData = 0;
  uint64_t insufficientSamples = 0;
  uint64_t insufficientPrecision = 0;
  uint64_t converged = 0;

  for (size_t i = 0; i < total; ++i) {
    const auto &d = dispatches[i];

    if (!d.time.has_value()) {
      noData += 1;
      insufficientSamples += 1;
      insufficientPrecision += 1;
      continue;
    }

    const auto &t = *d.time;
    const uint64_t n = t.samples.size();

    // --- minSamples check ---
    bool samplesOk = (n >= options.minSamples);
    if (!samplesOk)
      insufficientSamples += 1;

    // --- relative SEM check ---
    bool precisionOk = false;
    if (n > 1 && t.mean_latency_ns > 0 && t.std_derivation_ns > 0) {
      double sem_ns = static_cast<double>(t.std_derivation_ns) /
                      std::sqrt(static_cast<double>(n));

      double relError = sem_ns / static_cast<double>(t.mean_latency_ns);

      precisionOk = (relError <= static_cast<double>(options.maxRelativeError));
    }

    if (!precisionOk)
      insufficientPrecision += 1;

    if (samplesOk && precisionOk)
      converged += 1;
  }

  float progress =
      static_cast<float>(converged) / static_cast<float>(total) * 100.0f;

  logger.info("[{:>3}%] converged: {} / {} | "
              "minSamples pending: {} | "
              "precision pending: {} | "
              "no data: {}",
              static_cast<uint64_t>(std::floor(progress)), converged, total,
              insufficientSamples, insufficientPrecision, noData);

  return converged == total;
}

void denox::runtime::Db::bench(const DbBenchOptions &options) {
  assert(options.minSamples >= 1);

  BenchmarkState state = create_benchmark_state(m_context);

  diag::Logger logger("runtime.bench.db", true);

  bool done = print_progress_report(m_db, options, logger);
  if (done) {
    return;
  }

  memory::vector<uint32_t> iota(m_db.dispatches().size());
  std::iota(iota.begin(), iota.end(), 0);

  auto deviceProperties =
      vk::PhysicalDevice{m_context->vkPhysicalDevice()}.getProperties();

  std::string device_name = deviceProperties.deviceName;
  std::string os = denox::os_name();
  std::string driver_version =
      fmt::format("{}", deviceProperties.driverVersion);
  std::string denox_version = denox::version();
  std::string denox_commit_hash = denox::commit_hash();

  DbClockMode clock_mode = DbClockMode::Unavailable;

  uint64_t start_timestamp = benchmark_timestamp();

  uint32_t env = m_db.create_bench_environment(
      device_name, os, driver_version, denox_version, denox_commit_hash,
      start_timestamp, clock_mode, L2_WARMUP_ITERATONS, JIT_WARMUP_ITERATIONS,
      1);

  // benchmark configuration.
  const uint32_t epochSize = 500;
  const uint32_t maxBatchSize = 100;

  Epoch epochs[ASYNC_EPOCH_DEPTH]{};
  EpochBenchResults results[ASYNC_EPOCH_DEPTH];

  // cleanup
  std::array<std::atomic<bool>, ASYNC_EPOCH_DEPTH> epoch_is_live{};
  std::array<std::atomic<bool>, ASYNC_EPOCH_DEPTH> result_is_live{};

  // producer-consumer: epoch-creation -> benchmarking@main
  std::counting_semaphore emptyEpochs(ASYNC_EPOCH_DEPTH);
  std::counting_semaphore constructedEpochs(0);

  // producer-consumer: benchmarking@main -> dbwriteback
  std::counting_semaphore emptyResults(ASYNC_EPOCH_DEPTH);
  std::counting_semaphore fullResults(0);

  std::stop_source stop;
  auto epoch_creation = std::thread(
      [&](std::stop_token token) {
        std::vector<uint64_t> samples_in_flight(m_db.dispatches().size(), 0);
        const uint64_t big_sample_count = epochSize;

        uint32_t stage = 0;
        while (!token.stop_requested()) {
          emptyEpochs.acquire();
          // fmt::println("[epoch-construction] working");

          if (epoch_is_live[stage]) {
            epoch_is_live[stage].store(false); // <- don't access me anymore
            for (uint32_t target : epochs[stage].targets) {
              samples_in_flight[target] -= big_sample_count;
            }
            destroy_epoch(m_context, std::move(epochs[stage]));
          }

          memory::vector<uint32_t> selected_targets =
              select_targets_from_candidates(
                  state, m_db, iota, epochSize, options.minSamples,
                  options.maxRelativeError, true, samples_in_flight);

          if (selected_targets.empty()) {
            constructedEpochs.release(); // <- produce unalive epoch (signal)
            break;
          }

          uint32_t batchSize =
              static_cast<uint32_t>(selected_targets.size()); // heuristic!

          bool rme_convergence_mode = true;
          for (uint32_t target : selected_targets) {
            if (!m_db.dispatches()[target].time.has_value() ||
                m_db.dispatches()[target].time->samples.size() <
                    options.minSamples) {
              rme_convergence_mode = false;
            }
            samples_in_flight[target] += big_sample_count;
          }
          uint32_t sample_count = 0;
          if (rme_convergence_mode) {
            sample_count = maxBatchSize;
          } else {
            sample_count = batchSize * options.minSamples;
          }

          epoch_is_live[stage].store(true);
          epochs[stage] = create_epoch(m_context, m_db, selected_targets, env,
                                       batchSize, sample_count);
          assert(!epochs[stage].targets.empty());

          constructedEpochs.release();
          stage = (stage + 1) % ASYNC_EPOCH_DEPTH;
        }
        fmt::println("[epoch-construction] exit");
      },
      stop.get_token());

  auto dbwriteback = std::thread(
      [&](std::stop_token token) {
        uint32_t stage = 0;
        while (!token.stop_requested()) {
          fullResults.acquire();
          if (!result_is_live[stage].load()) {
            break;
          }

          const EpochBenchResults &result = results[stage];

          for (uint32_t i = 0; i < result.timings.size(); ++i) {
            const auto timing = result.timings[i];
            m_db.add_dispatch_benchmark_result(timing.target,
                                               std::move(timing.samples));
          }

          print_progress_report(m_db, options, logger);

          if (options.saveProgress) {
            m_db.atomic_writeback();
          }
          emptyResults.release();
          stage = (stage + 1) % ASYNC_EPOCH_DEPTH;
        }
        if (options.saveProgress) {
          m_db.atomic_writeback();
        }
        fmt::println("[writeback] exit");
      },
      stop.get_token());

  uint32_t stage = 0;
  bool first_iteration = true;
  bool warned_about_block = false;

  std::stop_token main_token = stop.get_token();
  while (!main_token.stop_requested()) {
    auto before_acquire = std::chrono::high_resolution_clock::now();
    constructedEpochs.acquire();
    auto acquire_took =
        std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
            std::chrono::high_resolution_clock::now() - before_acquire);
    if (!first_iteration &&
        acquire_took > std::chrono::duration<float, std::milli>(200.0f)) {
      if (!warned_about_block) {
        DENOX_WARN("main thread waited for {} for pipeline creation! Enable "
                   "parallel pipeline construction",
                   acquire_took);
        // warned_about_block = true;
      }
    }

    emptyResults.acquire();

    if (!epoch_is_live[stage].load()) {
      result_is_live[stage] = false;
      results[stage].timings.clear();
      fullResults.release(); // <- produce empty results (signal)
      break;
    }

    const Epoch &epoch = epochs[stage];

    try {
      assert(!epoch.targets.empty());
      result_is_live[stage] = true; // <- mark result as live!
      results[stage] = bench_epoch(state, m_db, m_context, epoch, options,
                                   epoch.sample_count);
    } catch (const std::exception &e) {
      // destroy_epoch(m_context, epoch);
      logger.error("{}Fatal exception exiting, without writeback\n{}{}",
                   logger.red(), e.what(), logger.reset());
      break;
    }
    fullResults.release();
    emptyEpochs.release();
    stage = (stage + 1) % ASYNC_EPOCH_DEPTH;
    first_iteration = false;
  }

  epoch_creation.join();
  dbwriteback.join();

  for (uint32_t i = 0; i < ASYNC_EPOCH_DEPTH; ++i) {
    if (epoch_is_live[i].load()) {
      epoch_is_live[i] = false;
      destroy_epoch(m_context, std::move(epochs[i]));
      epochs[i].targets.clear();
    }
  }
}

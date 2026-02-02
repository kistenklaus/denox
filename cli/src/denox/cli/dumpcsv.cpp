#include "dumpcsv.hpp"
#include "denox/cli/io/IOEndpoint.hpp"
#include "denox/cli/io/OutputStream.hpp"
#include "denox/common/commit_hash.hpp"
#include "denox/db/Db.hpp"

void dumpcsv(DumpCsvAction &action) {

  assert(action.database.endpoint.kind() == IOEndpointKind::Path);

  auto db = denox::Db::open(action.database.endpoint.path());

  std::vector<std::string> headers = {
      // Together identify a operation
      "operation",     //
      "input_shape",   //
      "input_format",  //
      "input_type",    //
      "output_shape",  //
      "output_format", //
      "output_type",   //
      // Together identify a implementation
      "shader", //
      "config", //
      // Implementation meta data.
      "spirv_size",       //
      "spirv_hash",       //
      "src_hash",         //
      "coopmat",          //
      "parameter_size",   //
      "descriptor_count", //
      "flops",            //
      "memory_reads",     //
      "memory_writes",    //
      // Measuring environment
      "device",          //
      "os",              //
      "driver_version",  //
      "start_timestamp", //
      "clock_mode",
      "l2_warmup_iterations",
      "jit_warmup_iterations",
      "measurement_iterations",
      // Timing
      "latency_ms", //
      "sample_timestamp",
  };

  std::string csv;
  csv.reserve(1 << 12);
  bool first = true;
  for (const auto &head : headers) {
    if (!first) {
      csv.push_back(',');
    }
    csv += head;
    first = false;
  }
  csv.push_back('\n');
  for (const denox::DbComputeDispatch &dispatch : db.dispatches()) {
    if (!dispatch.operation) {
      fmt::println("skipping line");
      continue; // skip lines without debug info.
    }
    if (!dispatch.input_bindings) {
      fmt::println("skipping line");
      continue;
    }
    if (!dispatch.output_bindings) {
      fmt::println("skipping line");
      continue;
    }
    bool skip = false;
    std::string input_shape;
    std::string input_format;
    std::string input_type;
    for (uint32_t i : *dispatch.input_bindings) {
      const auto &binding = dispatch.bindings[i];
      if (!binding.width || !binding.height || !binding.channels ||
          !binding.type) {
        skip = true;
        break;
      }
      uint32_t H = *binding.height;
      uint32_t W = *binding.width;
      uint32_t C = *binding.channels;
      input_shape = fmt::format("{}x{}x{}", H, W, C);
      input_format = fmt::format("{}", binding.format);
      input_type = fmt::format("{}", *binding.type);
    }
    if (skip) {

      fmt::println("skipping line");
      continue;
    }

    std::string output_shape;
    std::string output_format;
    std::string output_type;
    for (uint32_t o : *dispatch.output_bindings) {
      const auto &binding = dispatch.bindings[o];
      if (!binding.width || !binding.height || !binding.channels ||
          !binding.type) {
        skip = true;
        break;
      }
      uint32_t H = *binding.height;
      uint32_t W = *binding.width;
      uint32_t C = *binding.channels;
      output_shape = fmt::format("{}x{}x{}", H, W, C);
      output_format = fmt::format("{}", binding.format);
      output_type = fmt::format("{}", *binding.type);
    }
    if (skip) {
      fmt::println("skipping line");
      continue;
    }

    if (!dispatch.config || !dispatch.shader_name) {
      fmt::println("skipping line");
      continue;
    }
    std::string shader = *dispatch.shader_name;
    std::string config = *dispatch.config;

    const auto binary = db.binaries()[dispatch.binaryId];
    const denox::SHA256 &src_hash = binary.hash;
    uint64_t spv_size = binary.spvBinary.spv.size() * sizeof(uint32_t);

    denox::SHA256Builder spv_hasher;
    spv_hasher.update(
        {reinterpret_cast<const uint8_t *>(binary.spvBinary.spv.data()),
         spv_size});
    denox::SHA256 spv_hash = spv_hasher.finalize();
    bool coopmat = dispatch.coopmat.value_or(false);
    size_t parameterSize = 0;
    for (const auto &binding : dispatch.bindings) {
      if (binding.is_param) {
        parameterSize += binding.byteSize;
      }
    }
    size_t descriptorCount = dispatch.bindings.size();

    uint64_t flops = dispatch.flops.value_or(0);
    uint64_t memory_reads = dispatch.memory_reads.value_or(0);
    uint64_t memory_writes = dispatch.memory_writes.value_or(0);

    std::string line_header = fmt::format(
        "\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{:40}\",\"{:40}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"",
        *dispatch.operation,
        input_shape, input_format, input_type, output_shape, output_format,
        output_type, shader, config, spv_size, spv_hash, src_hash, coopmat,
        parameterSize, descriptorCount, flops, memory_reads, memory_writes);

    if (!dispatch.time) {
      continue;
    }

    const denox::DbDispatchTiming &timing = *dispatch.time;

    for (const denox::DbSample &sample : timing.samples) {
      const denox::DbEnv &env = db.envs()[sample.env];

      std::string clock_mode;
      switch (env.clock_mode) {
      case denox::DbClockMode::Unavailable:
        clock_mode = "unavailable";
        break;
      case denox::DbClockMode::None:
        clock_mode = "none";
        break;
      case denox::DbClockMode::Base:
        clock_mode = "base";
        break;
      case denox::DbClockMode::Maximum:
        clock_mode = "max";
        break;
      }

      csv += fmt::format(
          "{},\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\"\n", line_header, env.device, env.os,
          env.driver_version, env.start_timestamp, clock_mode,
          env.l2_warmup_iterations, env.jit_warmup_iterations,
          env.measurement_iterations,
          static_cast<float>(sample.latency_ns) * 1e-6f, sample.timestamp);
    }
  }

  OutputStream{action.csv}.write_exact(
      std::span{reinterpret_cast<const std::byte *>(csv.data()), csv.size()});
}

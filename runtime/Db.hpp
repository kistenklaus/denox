#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace denox::runtime {

struct ShaderBinary {
  std::vector<uint32_t> source;
};

enum class Access {
  ReadOnly,
  WriteOnly,
  ReadWrite,
};

struct TensorBinding {
  unsigned int set;
  unsigned int binding;
  Access access;

  uint64_t byteSize;
  unsigned int minAlignment;
};

struct Timing {
  uint64_t samples = 0;
  uint64_t latency_ns = 0;
  uint64_t std_derivation_ns = 0;
};

struct ComputeDispatch {
  uint32_t binaryId;
  uint32_t workgroupCountX;
  uint32_t workgroupCountY;
  uint32_t workgroupCountZ;
  std::vector<uint8_t> pushConstant;
  std::vector<TensorBinding> bindings;
  Timing time;
};

struct Operation {
  std::string shaderName;
  uint32_t pattern;
  uint32_t config;
  uint64_t hash;
  std::vector<uint32_t> dispatches;
};

struct Db {
  static Db open(const std::string &path);

  std::string path;
  std::vector<ShaderBinary> binaries;
  std::vector<ComputeDispatch> dispatches;
  std::vector<Operation> operations;

  void write_back();
};

} // namespace denox::runtime

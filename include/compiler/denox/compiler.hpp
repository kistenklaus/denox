#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace denox {

enum class SrcType {
  Auto, // <- infer from file extention.
  ONNX, // <- force onnx frontend.
};

enum class CoopMatElemType {
  float16_t,
};

enum class CoopMatScope {
  Device,
  QueueFamily,
  Workgroup,
  Subgroup,
};

struct DeviceCoopMatType {
  unsigned int m;
  unsigned int k;
  unsigned int n;
  CoopMatElemType aType;
  CoopMatElemType bType;
  CoopMatElemType accType;
  CoopMatScope scope;
};

struct DeviceInfo {
  std::pair<unsigned int, unsigned int> spirvVersion;
  DeviceCoopMatType *coopmatTypes;
  std::size_t coopmatTypeCount;
};

enum FeatureState {
  Require,
  Enable,
  Disable,
};

struct Features {
  FeatureState fusion = Enable;
  FeatureState memory_concat = Enable;
  FeatureState coopmat = Enable;
};

struct Heuristic {
  // TODO later (but a discriminated type union or something like it.)
};

struct CompileOptions {
  unsigned int version = 0; // <- 0 no specific version.
  SrcType srcType = SrcType::Auto;
  DeviceInfo *deviceInfo = nullptr;
  Features features;
  Heuristic heuristic = {};
  const char *cwd = nullptr;
};

void compile(const char *path, const CompileOptions &options = {});
void compile(void *data, std::size_t n, const CompileOptions &options = {});

} // namespace denox

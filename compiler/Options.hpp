#pragma once

#include "io/fs/Path.hpp"
#include "memory/container/optional.hpp"
#include "memory/container/vector.hpp"
#include "memory/dtype/dtype.hpp"
namespace denox::compiler {

enum class SrcType {
  Onnx,
};

struct CoopMatType {
  unsigned int m;
  unsigned int k;
  unsigned int n;
  memory::Dtype atype;
  memory::Dtype btype;
  memory::Dtype accType;
};

struct DeviceInfo {
  std::pair<unsigned int, unsigned int> spirvVersion;
  memory::vector<CoopMatType> coopmatTypes;
};

struct Features {
  bool fusion;
  bool memory_concat;
  bool coopmat;
};

struct Options {
  unsigned int version;
  SrcType srcType;
  DeviceInfo deviceInfo;
  Features features;
  io::Path cwd;
  memory::optional<io::Path> srcPath;
};

} // namespace denox::compiler

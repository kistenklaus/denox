#pragma once

#include <cstdint>
namespace denox {

enum class Storage {
  StorageBuffer,
};

enum class Layout {
  Undefined,
  HWC,
  CHW,
  CHWC8,
};

enum DataType {
  Auto,
  Float16,
  Float32,
  Uint8,
  Int8,
};

struct Extent {
  const char *name;
  uint64_t value;
};

struct Shape {
  Extent height;
  Extent width;
  Extent channels;
};

struct BufferDescription {
  Shape shape;
  Storage storage;
  Layout layout;
  DataType dtype;
};

}

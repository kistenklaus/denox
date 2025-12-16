#pragma once

#include <cstdint>
namespace denox {

enum class VulkanApiVersion {
  Vulkan_1_0,
  Vulkan_1_1,
  Vulkan_1_2,
  Vulkan_1_3,
  Vulkan_1_4,
};


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

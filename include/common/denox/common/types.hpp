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
  StorageImage,
  Sampler,
};

enum class Layout {
  Undefined = 0,
  HWC,
  CHW,
  CHWC8,
  RGBA,
  RGB,
  RG,
  R
};

enum DataType {
  Auto,
  Float16,
  Float32,
  Uint8,
  Int8,
};

struct Extent {
  const char *name = nullptr;
  uint64_t value = 0;
};

struct Shape {
  Extent height;
  Extent width;
  Extent channels;
};

struct BufferDescription {
  char* name;
  Shape shape;
  Storage storage = Storage::StorageBuffer;
  Layout layout = Layout::Undefined;
  DataType dtype = DataType::Auto;
};

}

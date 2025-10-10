#pragma once

#include <cstddef>
#include <cstdlib>

namespace denox {

struct Extent {
  unsigned int width;
  unsigned int height;
  unsigned int channels;
};

enum class Layout {
  HWC,
  CHW,
  CHWC8,
};

enum class Dtype {
  F16,
};

struct Tensor {
  void *data;
  Extent extent;
  Layout layout;
  Dtype dtype;
};

typedef void* RuntimeContext;
typedef void *RuntimeModel;
typedef void *RuntimeBuffer;

int create_runtime_context(const char* deviceName, RuntimeContext* context);
void destroy_runtime_context(RuntimeContext context);

int create_runtime_model(RuntimeContext context, const void *dnx, size_t dnxSize, RuntimeModel *model);
void destroy_runtime_model(RuntimeContext context, RuntimeModel model);

int create_input_buffers(RuntimeModel model, int intputCount, Extent *extents,
                         RuntimeBuffer **inputs);

int create_output_buffers(RuntimeModel model, RuntimeBuffer *inputs,
                          int outputCount, RuntimeBuffer **outputs);

void destroy_buffers(RuntimeModel model, int count, RuntimeBuffer *buffers);

int eval(RuntimeModel model, denox::RuntimeBuffer *inputs,
         RuntimeBuffer *outputs);

size_t get_buffer_size(RuntimeModel model, RuntimeBuffer buffer);
Extent get_buffer_extent(RuntimeModel model, RuntimeBuffer buffer);
Dtype get_buffer_dtype(RuntimeModel model, RuntimeBuffer buffer);
Layout get_buffer_layout(RuntimeModel model, RuntimeBuffer buffer);

void *map_buffer(RuntimeModel model, RuntimeBuffer buffer);
void unmap_buffer(RuntimeModel model, RuntimeBuffer buffer);

} // namespace denox


#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace denox {

struct DynamicExtent {
  uint64_t value;
  const char *name;
};

typedef void *RuntimeContext;
typedef void *RuntimeModel;
typedef void *RuntimeInstance;

typedef void *RuntimeBuffer;

int create_runtime_context(const char *deviceName, RuntimeContext *context);
void destroy_runtime_context(RuntimeContext context);

int create_runtime_model(RuntimeContext context, const void *dnx,
                         size_t dnxSize, RuntimeModel *model);
void destroy_runtime_model(RuntimeContext context, RuntimeModel model);

int create_runtime_instance(RuntimeContext context, RuntimeModel model,
                            int dynamicExtentCount,
                            DynamicExtent *dynamicExtents,
                            RuntimeInstance *instance);

void destroy_runtime_instance(RuntimeContext context, RuntimeInstance instance);

int eval_runtime_instance(RuntimeContext context, RuntimeInstance instance,
                          void **inputs, void **outputs);

int get_runtime_model_input_count(RuntimeModel model);

int get_runtime_model_output_count(RuntimeInstance model);

int get_runtime_instance_extent(RuntimeInstance instance,
                                const char *extentName);

int get_runtime_instance_tensor_shape(RuntimeInstance instance,
                                      const char *tensorName,
                                      std::uint32_t *height,
                                      std::uint32_t *width,
                                      std::uint32_t *channels);

std::size_t get_runtime_instance_tensor_byte_size(RuntimeInstance instance,
                                                  const char *tensorName);

} // namespace denox

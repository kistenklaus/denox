#pragma once

#include "denox/common/types.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace denox {

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
                            int dynamicExtentCount, Extent *dynamicExtents,
                            RuntimeInstance *instance);

void destroy_runtime_instance(RuntimeContext context, RuntimeInstance instance);

int eval_runtime_instance(RuntimeContext context, RuntimeInstance instance,
                          void **inputs, void **outputs);

int get_runtime_model_input_count(RuntimeModel model);
int get_runtime_model_output_count(RuntimeModel model);

const char *get_runtime_model_input_name(RuntimeModel model, int index);
const char *get_runtime_model_output_name(RuntimeModel model, int index);

std::uint64_t get_runtime_instance_extent(RuntimeInstance instance,
                                          const char *extentName);

int get_runtime_instance_tensor_shape(RuntimeInstance instance,
                                      const char *tensorName, Extent *height,
                                      Extent *width, Extent *channels);

std::size_t get_runtime_instance_tensor_byte_size(RuntimeInstance instance,
                                                  const char *tensorName);

DataType get_runtime_model_tensor_dtype(RuntimeModel model,
                                        const char *tensorName);

Layout get_runtime_model_tensor_layout(RuntimeModel model,
                                       const char *tensorName);

} // namespace denox

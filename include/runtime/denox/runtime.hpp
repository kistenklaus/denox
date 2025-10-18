#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace denox {

struct DynamicExtent {
  uint64_t value;
  const char *name;
};

struct EvalResult {
  int outputCount;
  void **outputs;
};

typedef void *RuntimeContext;
typedef void *RuntimeModel;
typedef void *RuntimeModelInstance;

typedef void *RuntimeBuffer;

int create_runtime_context(const char *deviceName, RuntimeContext *context);
void destroy_runtime_context(RuntimeContext context);

int create_runtime_model(RuntimeContext context, const void *dnx,
                         size_t dnxSize, RuntimeModel *model);
void destroy_runtime_model(RuntimeContext context, RuntimeModel model);

int create_runtime_model_instance(RuntimeContext context, RuntimeModel model,
                                  int dynamicExtentCount,
                                  DynamicExtent *dynamicExtents,
                                  RuntimeModelInstance *instance);

void destroy_runtime_model_instance(RuntimeContext context,
                                    RuntimeModelInstance instance);

int eval_runtime_model_instance(RuntimeContext context,
                                RuntimeModelInstance instance, 
                                void **inputs, void** outputs);

int get_runtime_model_input_count(RuntimeModel model);

int get_runtime_model_output_count(RuntimeModel model);

void get_runtime_model_instance_input_shape(RuntimeModelInstance instance,
                                            int input, size_t *width,
                                            size_t *height, size_t *channels);

void get_runtime_model_instance_input_byte_size(RuntimeModelInstance instance,
                                                int input, size_t *byteSize);

void get_runtime_model_instance_output_shape(RuntimeModelInstance instance,
                                             int output, size_t *width,
                                             size_t *height, size_t *channels);

void get_runtime_model_instance_output_byte_size(RuntimeModelInstance instance,
                                                 int coutput, size_t *byteSize);

} // namespace denox

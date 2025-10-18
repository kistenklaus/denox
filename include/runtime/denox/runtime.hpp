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
                                 RuntimeModelInstance instance, int inputCount,
                                 void **inputs, EvalResult *result);

void destroy_eval_result(EvalResult result);

} // namespace denox

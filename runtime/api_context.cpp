#include "context.hpp"
#include "denox/runtime.hpp"

namespace denox {

int create_runtime_context(const char *deviceName, RuntimeContext *context, VulkanApiVersion target_env) {
  assert(context);
  auto *ctx = new runtime::Context(deviceName, target_env);
  *context = static_cast<void *>(ctx);
  return 0;
}

void destroy_runtime_context(RuntimeContext context) {
  assert(context);
  auto *ctx = reinterpret_cast<runtime::Context *>(context);
  delete ctx;
}

} // namespace denox

#include "context.hpp"
#include "denox/runtime.hpp"

namespace denox {

int create_runtime_context(const char *deviceName, RuntimeContext *context) {
  assert(context);
  auto *ctx = new runtime::Context(deviceName);
  *context = static_cast<void *>(ctx);
  return 0;
}

void destroy_runtime_context(RuntimeContext context) {
  assert(context);
  auto *ctx = reinterpret_cast<runtime::Context *>(context);
  delete ctx;
}

} // namespace denox

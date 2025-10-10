#include "context.hpp"
#include "denox/runtime.hpp"
#include "dnx.h"
#include "model.hpp"
#include <fmt/printf.h>
#include <stdexcept>

namespace denox {

int create_runtime_context(const char *deviceName, RuntimeContext *context) {
  auto *ctx = new runtime::Context(deviceName);
  *context = static_cast<void *>(ctx);
  return 0;
}

void destroy_runtime_context(RuntimeContext context) {
  auto *ctx = reinterpret_cast<runtime::Context *>(context);
  delete ctx;
}

int create_runtime_model(RuntimeContext context, const void *dnx,
                         size_t dnxSize, RuntimeModel *out_model) {
  flatbuffers::Verifier verifier(static_cast<const std::uint8_t *>(dnx),
                                 dnxSize);
  if (!denox::dnx::VerifyModelBuffer(verifier)) {
    throw std::runtime_error("Failed to verify dnx file format.");
  }
  void *dnxBuffer = malloc(dnxSize);
  std::memcpy(dnxBuffer, dnx, dnxSize);
  const dnx::Model *dnxModel = dnx::GetModel(dnxBuffer);
  auto *model = new runtime::Model(dnxBuffer, dnxModel);

  return 0;
}

void destroy_runtime_model(RuntimeContext context, RuntimeModel model) {}

int create_input_buffers(RuntimeModel model, int intputCount, Extent *extents,
                         RuntimeBuffer **inputs) {
  return -1;
}

int create_output_buffers(RuntimeModel model, RuntimeBuffer *inputs,
                          int outputCount, RuntimeBuffer **outputs) {
  return -1;
}

void destroy_buffers(RuntimeModel model, int count, RuntimeBuffer *buffers) {}

int eval(RuntimeModel model, denox::RuntimeBuffer *inputs,
         RuntimeBuffer *outputs) {
  return -1;
}

size_t get_buffer_size(RuntimeModel model, RuntimeBuffer buffer) { return -1; }

Extent get_buffer_extent(RuntimeModel model, RuntimeBuffer buffer) {
  Extent extent;
  return extent;
}

Dtype get_buffer_dtype(RuntimeModel model, RuntimeBuffer buffer) {
  Dtype dtype;
  return dtype;
}

Layout get_buffer_layout(RuntimeModel model, RuntimeBuffer buffer) {
  Layout layout;
  return layout;
}

void *map_buffer(RuntimeModel model, RuntimeBuffer buffer) { return nullptr; }

void unmap_buffer(RuntimeModel model, RuntimeBuffer buffer) {}

} // namespace denox

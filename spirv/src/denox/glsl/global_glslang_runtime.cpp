#include "denox/glsl/global_glslang_runtime.hpp"
#include <glslang/Public/ShaderLang.h>
#include <mutex>
#include <stdexcept>

namespace denox::glslang {

static bool runtime_is_initalized = false;
static bool externally_managed_runtime = false;
static bool runtime_was_finialized = false;
static std::mutex global_runtime_lock;

void ensure_initialized() {
  std::lock_guard lck{global_runtime_lock};
  if (runtime_is_initalized) {
    return;
  }
  if (runtime_was_finialized) {
    throw std::runtime_error("Trying to call glslang::InitalizeProcess twice!");
  }
  ::glslang::InitializeProcess();
  runtime_is_initalized = true;
}

void finalize() {
  std::lock_guard lck{global_runtime_lock};
  if (externally_managed_runtime) {
    return;
  }
  if (!runtime_is_initalized) {
    return;
  }
  if (runtime_was_finialized) {
    throw std::runtime_error("Trying to call glslang::FinalizeProcess twice!");
  }
  if (runtime_is_initalized) {
    ::glslang::FinalizeProcess();
  }
  runtime_was_finialized = true;
  runtime_is_initalized = false;
}

void assume_externally_managed() {
  std::lock_guard lck{global_runtime_lock};

  if (externally_managed_runtime) {
    return;
  }
  if (runtime_is_initalized) {
    throw std::runtime_error(
        "Trying to mark the global glsl runtime as externally managed, but "
        "runtime was already initalized");
  }

  externally_managed_runtime = true;
  runtime_is_initalized = true;
}

} // namespace denox::glslang

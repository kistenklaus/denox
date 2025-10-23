#include "RuntimeContext.hpp"
#include "RuntimeModel.hpp"

namespace pydenox {

RuntimeContext::RuntimeContext(py::object device) {
  std::string patternStore;
  const char *pattern = nullptr;
  if (!device.is_none()) {
    patternStore = py::cast<std::string>(device);
    pattern = patternStore.c_str();
  }
  py::gil_scoped_release release;
  if (!denox::create_runtime_context(pattern, &m_context)) {
    throw std::runtime_error("Failed to create denox runtime context");
  }
}

RuntimeContext::~RuntimeContext() { release(); }

RuntimeContext *RuntimeContext::enter() { return this; }

bool RuntimeContext::exit(py::object exc_type, py::object exc, py::object tb) {
  release();
  return false;
}

RuntimeModel
RuntimeContext::load_model(const py::object &program_or_bytes) const {
  throw std::runtime_error("RuntimeContext load_model is not implemented");
}

std::string RuntimeContext::repr() const {
  throw std::runtime_error("RuntimeContext repr is not implemented");
}

void RuntimeContext::release() {
  if (m_context != nullptr) {
    try {
      py::gil_scoped_release release;
      denox::destroy_runtime_context(m_context);
    } catch (...) {
      m_context = nullptr; // intentional leak!
    }
  }
  m_context = nullptr;
}

}

#include "Module.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include <filesystem>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <pybind11/stl.h>
#include <stdexcept>

namespace pydenox {

Module Module::load(pybind11::object pypath) {
  const std::string pathstr = pybind11::str(pypath);
  pybind11::gil_scoped_release release;
  namespace fs = std::filesystem;
  const fs::path path{pathstr};

  if (!fs::exists(path)) {
    throw std::runtime_error(fmt::format(
        "Failed to read model. File \"{}\" does not exist.", pathstr));
  }
  const std::uintmax_t size = fs::file_size(path);
  if (size == std::numeric_limits<std::uintmax_t>::max()) {
    throw std::runtime_error(
        fmt::format("Failed to determine file size of file: \"{}\"", pathstr));
  }
  void *buffer = malloc(size);
  std::ifstream file{path, std::ios::binary};
  if (!file) {
    throw std::runtime_error(
        fmt::format("Failed to open file: \"{}\"", pathstr));
  }
  file.read(static_cast<char *>(buffer), static_cast<std::streamsize>(size));
  if (!file) {
    throw std::runtime_error(
        fmt::format("Failed to read entire file: \"{}\"", pathstr));
  }

  return Module(buffer, size);
}

void Module::define(pybind11::module_ &m) {
  namespace py = pybind11;
  py::class_<Module>(m, "Module")
      .def_static(
          "load", &Module::load, py::arg("path"),
          "Read a DNX model from a file and return a new Model instance.")
      .def("__enter__", &Module::enter,
           py::return_value_policy::reference_internal,
           "Context manager entry. Returns self.")
      .def("__exit__", &Module::exit, py::arg("exc_type") = py::none(),
           py::arg("exc") = py::none(), py::arg("tb") = py::none(),
           "Context manager exit. Frees model resources.\n"
           "Does not suppress exceptions.")

      // --- File / Data Operations ---
      .def("save", &Module::save, py::arg("path"),
           "Write the model's bytes to a file.")
      .def("bytes", &Module::bytes,
           "Return the model as a Python `bytes` object (copies the data).")

      // --- Special Python Protocols ---
      .def("__bytes__", &Module::bytes)
      .def("__len__",
           [](const Module &self) {
             return static_cast<py::ssize_t>(self.size());
           })
      .def("__repr__", &Module::repr,
           "Developer-readable representation, e.g. "
           "<denox.Model size=32768 bytes>.")
      .def_static(
          "compile", &Module::compile, py::arg("model"), py::kw_only(),
          py::arg("device") = py::none(),
          py::arg("target_env") = denox::VulkanApiVersion::Vulkan_1_1,
          py::arg("input_shape") = pydenox::Shape(),
          py::arg("output_shape") = pydenox::Shape(),
          py::arg("input_type") = denox::DataType::Auto,
          py::arg("output_type") = denox::DataType::Auto,
          py::arg("input_layout") = denox::Layout::Undefined,
          py::arg("output_layout") = denox::Layout::Undefined,
          py::arg("input_storage") = denox::Storage::StorageBuffer,
          py::arg("output_storage") = denox::Storage::StorageBuffer,
          py::arg("coopmat") = py::none(), //
          py::arg("fusion") = py::none(),  //
          py::arg("memory_concat") = py::none(),
          py::arg("spirv_debug_info") = false,
          py::arg("spirv_non_semantic_debug_info") = false,
          py::arg("spirv_optimize") = true, py::arg("verbose") = false,
          py::arg("summary") = false, py::arg("quiet") = false,
          "Compile an ONNX model or file into a DNX runtime module.\n"
          "\n"
          "Args:\n"
          "  model: Either an ONNXProgram (in-memory) or a path to an .onnx "
          "file.\n"
          "  device: Device selector string, e.g. '*RTX*'.\n"
          "  target_env: Vulkan target version.\n"
          "  input_shape: Optional input shape (H, W, C) or symbolic extents.\n"
          "  output_shape: Optional output shape.\n"
          "  input_type/output_type: Data type strings ('f16', 'f32', 'auto', "
          "...).\n"
          "  input_layout/output_layout: Layout specifiers ('HWC', 'CHW', "
          "...).\n"
          "  input_storage/output_storage: Storage specifiers ('SSBO', ...).\n"
          "  coopmat/fusion/memory_concat: Feature flags (bool or string).\n"
          "  spirv_debug_info/spirv_non_semantic_debug_info: Enable debug "
          "info.\n"
          "  spirv_optimize: Run SPIR-V optimization passes.\n"
          "\n"
          "Returns:\n"
          "  A compiled DNX Module ready for runtime loading.")
      .def("__call__", &Module::infer,                                 //
           py::arg("input"), py::kw_only(),                            //
           py::arg("device") = py::none(),                             //
           py::arg("target_env") = denox::VulkanApiVersion::Vulkan_1_1 //
      );
}

Module::Module(void *dnxBuffer, std::size_t dnxBufferSize)
    : m_dnxBuffer(dnxBuffer), m_dnxBufferSize(dnxBufferSize) {}

Module::~Module() { release(); }

Module *Module::enter() { return this; }

bool Module::exit([[maybe_unused]] pybind11::object exc_type,
                  [[maybe_unused]] pybind11::object exc,
                  [[maybe_unused]] pybind11::object tb) {
  release();
  return false;
}

void Module::save(pybind11::object pypath) {
  if (m_dnxBuffer == nullptr) {
    throw std::runtime_error("Model is empty");
  }
  const std::string path = pybind11::str(pypath);
  pybind11::gil_scoped_release release;
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error(fmt::format("Failed to open file: \"{}\"", path));
  }
  file.write(static_cast<const char *>(m_dnxBuffer),
             static_cast<std::streamsize>(m_dnxBufferSize));
  if (!file) {
    throw std::runtime_error(fmt::format("Failed to write file: {}", path));
  }
}

pybind11::bytes Module::bytes() const {
  if (m_dnxBuffer == nullptr) {
    return pybind11::bytes();
  }
  return pybind11::bytes(static_cast<const char *>(m_dnxBuffer),
                         m_dnxBufferSize);
}

std::string Module::repr() const {
  return fmt::format("<denox.Model size={} bytes>", m_dnxBufferSize);
}

std::size_t Module::size() const { return m_dnxBufferSize; }

const void *Module::get() const { return m_dnxBuffer; }

void Module::release() {
  if (m_dnxBuffer != nullptr) {
    free(m_dnxBuffer);
    m_dnxBufferSize = 0;
    m_dnxBuffer = nullptr;
  }
  assert(m_dnxBuffer == nullptr);
  assert(m_dnxBufferSize == 0);
}

} // namespace pydenox

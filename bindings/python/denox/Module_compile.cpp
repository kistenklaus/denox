#include "Module.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include <fmt/base.h>
#include <fmt/format.h>
#include <pybind11/stl.h>
#include <span>
#include <stdexcept>

namespace pydenox {

// Store names here so `const char*` lifetimes cover the compile() call.
struct NameArena {
  std::list<std::string>
      pool; // node-based → insertions don't move existing nodes
  const char *intern(std::string s) {
    pool.emplace_back(std::move(s));
    return pool.back().c_str(); // stable while the list element lives
  }
};

static denox::Extent convert_extent(const pybind11::object &o,
                                    NameArena &arena) {
  denox::Extent e{nullptr, 0};
  if (o.is_none())
    return e;

  if (pybind11::isinstance<pybind11::int_>(o)) {
    e.value = static_cast<uint64_t>(o.cast<uint64_t>());
    return e;
  }

  if (pybind11::isinstance<pybind11::str>(o)) {
    std::string s = o.cast<std::string>();
    if (!s.empty())
      e.name = arena.intern(std::move(s));
    return e;
  }

  throw std::invalid_argument("Shape field must be int, str, or None");
}

static denox::Shape convert_shape(const pydenox::Shape &s, NameArena &arena) {
  denox::Shape out{};
  out.height = convert_extent(s.height, arena);
  out.width = convert_extent(s.width, arena);
  out.channels = convert_extent(s.channels, arena);
  return out;
}

Module Module::compile(pybind11::object model,
                       // kwargs
                       std::optional<std::string> device,  //
                       denox::VulkanApiVersion target_env, //
                       pydenox::Shape input_shape,         //
                       pydenox::Shape output_shape,        //
                       denox::DataType input_type,         //
                       denox::DataType output_type,        //
                       denox::Layout input_layout,         //
                       denox::Layout output_layout,        //
                       denox::Storage input_storage,       //
                       denox::Storage output_storage,      //
                       std::optional<bool> coopmat,        //
                       std::optional<bool> fusion,         //
                       std::optional<bool> memory_concat,  //
                       bool spirv_debug_info,              //
                       bool spirv_non_semantic_debug_info, //
                       bool spirv_optimize,                //
                       bool verbose, bool summary, bool quiet) {
  NameArena arena;
  denox::CompileOptions opt{};
  opt.device.deviceName = device ? arena.intern(*device) : nullptr;
  opt.device.apiVersion = target_env;

  opt.dnxVersion = 0;
  opt.srcType = denox::SrcType::Auto;
  opt.heuristic = denox::Heuristic::MemoryBandwidth;

  opt.features.coopmat = coopmat.has_value()
                             ? (*coopmat ? denox::FeatureState::Disable
                                         : denox::FeatureState::Require)
                             : denox::FeatureState::Enable;
  opt.features.fusion = fusion.has_value()
                            ? (*fusion ? denox::FeatureState::Disable
                                       : denox::FeatureState::Require)
                            : denox::FeatureState::Enable;
  opt.features.memory_concat =
      memory_concat.has_value()
          ? (*memory_concat ? denox::FeatureState::Disable
                            : denox::FeatureState::Require)
          : denox::FeatureState::Enable;

  opt.spirvOptions.debugInfo = spirv_debug_info;
  opt.spirvOptions.nonSemanticDebugInfo = spirv_non_semantic_debug_info;
  opt.spirvOptions.optimize = spirv_optimize;
  opt.spirvOptions.skipCompilation = false;

  opt.verbose = verbose;
  opt.quiet = quiet;
  opt.summarize = summary;

  opt.cwd = nullptr;

  opt.inputDescription.shape = convert_shape(input_shape, arena);
  opt.outputDescription.shape = convert_shape(output_shape, arena);
  opt.inputDescription.dtype = input_type;
  opt.outputDescription.dtype = output_type;
  opt.inputDescription.layout = input_layout;
  opt.outputDescription.layout = output_layout;
  opt.inputDescription.storage = input_storage;
  opt.outputDescription.storage = output_storage;

  // TODO parse options. later.

  denox::CompilationResult result;
  if (pybind11::isinstance<pybind11::str>(model)) {
    // Case 1: path to ONNX file
    std::string path = pybind11::cast<std::string>(model);
    pybind11::gil_scoped_release release;
    if (denox::compile(path.c_str(), &opt, &result) < 0)
      throw std::runtime_error(
          fmt::format("Failed to compile:\n{}", result.message));
  } else if (pybind11::hasattr(model, "save")) {
    opt.srcType = denox::SrcType::Onnx;
    // Case 2: ONNXProgram
    pybind11::object io = pybind11::module_::import("io");
    pybind11::object buf = io.attr("BytesIO")();
    model.attr("save")(buf);
    pybind11::object data = buf.attr("getvalue")();

    pybind11::buffer b = data.cast<pybind11::buffer>();
    pybind11::buffer_info info = b.request();
    void *ptr = info.ptr;
    std::size_t size = info.size * info.itemsize;

    void *optr = std::malloc(size);
    std::memcpy(optr, ptr, size);

    pybind11::gil_scoped_release release;
    if (denox::compile(optr, size, &opt, &result) < 0) {
      std::free(optr);
      throw std::runtime_error(
          fmt::format("Failed to compile:\n{}", result.message));
    }
    std::free(optr);
  } else if (pybind11::hasattr(model, "__buffer__") ||
             pybind11::hasattr(model, "to_bytes") ||
             pybind11::isinstance<pybind11::bytes>(model) ||
             pybind11::isinstance<pybind11::bytearray>(model)) {
    opt.srcType = denox::SrcType::Onnx;
    // Case 3: bytes-like object
    pybind11::object data = model;
    if (pybind11::hasattr(model, "to_bytes"))
      data = model.attr("to_bytes")();
    pybind11::buffer b = data.cast<pybind11::buffer>();
    pybind11::buffer_info info = b.request();
    void *ptr = info.ptr;
    std::size_t size = info.size * info.itemsize;

    void *optr = std::malloc(size);
    std::memcpy(optr, ptr, size);

    pybind11::gil_scoped_release release;
    if (denox::compile(optr, size, &opt, &result) < 0) {
      std::free(optr);
      throw std::runtime_error(
          fmt::format("Failed to compile:\n{}", result.message));
    }
    std::free(optr);
  } else {
    throw std::invalid_argument(
        "model must be a path, ONNXProgram (with .save), or bytes-like object");
  }
  void *dnx = malloc(result.dnxSize);
  std::size_t size = result.dnxSize;
  std::memcpy(dnx, result.dnx, size);

  denox::destroy_compilation_result(&result);

  return Module{dnx, size};
}

} // namespace pydenox

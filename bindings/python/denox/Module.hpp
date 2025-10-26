#pragma once

#include "Shape.hpp"
#include "Tensor.hpp"
#include "denox/compiler.hpp"
#include <cstddef>
#include <fmt/base.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace pydenox {

class Module {
public:
  static Module load(pybind11::object pypath);
  static void define(pybind11::module_ &m);
  static Module compile(pybind11::object model,
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
                        bool verbose, bool summary, bool quiet);
  Module(const Module &) = default;            // shallow copy → double free
  Module &operator=(const Module &) = default; // shallow copy → double free

  Module(Module &&o)
      : m_dnxBuffer(std::exchange(o.m_dnxBuffer, nullptr)),
        m_dnxBufferSize(std::exchange(o.m_dnxBufferSize, 0)) {}
  Module &operator=(Module &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    m_dnxBuffer = std::exchange(o.m_dnxBuffer, nullptr);
    m_dnxBufferSize = std::exchange(o.m_dnxBufferSize, 0);

    return *this;
  }

  explicit Module(void *dnxBuffer, std::size_t dnxBufferSize);
  ~Module();
  Module *enter();
  bool exit(pybind11::object exc_type, pybind11::object exc,
            pybind11::object tb);
  void save(pybind11::object pypath);
  pybind11::bytes bytes() const;
  std::string repr() const;
  std::size_t size() const;
  const void *get() const;

  pybind11::object infer(pybind11::object input,
                         // kwargs
                         std::optional<std::string> device,
                         denox::VulkanApiVersion targetEnv) {

    Tensor tensor =
        Tensor::from(input, denox::DataType::Auto, denox::Layout::CHW);
    fmt::println("C = {}", tensor.channels());
    fmt::println("H = {}", tensor.width());
    fmt::println("w = {}", tensor.height());

    tensor = tensor.transform(denox::DataType::Float16, denox::Layout::HWC);
    // TODO (Ignore for now)

    throw std::runtime_error("not-implemented");
  }

private:
  void release();

private:
  void *m_dnxBuffer;
  std::size_t m_dnxBufferSize;
};

} // namespace pydenox

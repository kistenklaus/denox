#pragma once

#include "ContextManager.hpp"
#include "Shape.hpp"
#include "Tensor.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include "denox/runtime.hpp"
#include <cstddef>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace pydenox {

class Module {
private:
  struct RuntimeModelDeleter {
    void operator()(denox::RuntimeModel *ptr) const {}
    denox::RuntimeContext ctx;
  };

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
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;

  Module(Module &&o)
      : m_dnxBuffer(std::exchange(o.m_dnxBuffer, nullptr)),
        m_dnxBufferSize(std::exchange(o.m_dnxBufferSize, 0)),
        m_contextManager(std::move(o.m_contextManager)) {}
  Module &operator=(Module &&o) {
    if (this == &o) {
      return *this;
    }
    release();
    m_dnxBuffer = std::exchange(o.m_dnxBuffer, nullptr);
    m_dnxBufferSize = std::exchange(o.m_dnxBufferSize, 0);
    std::swap(o.m_contextManager, o.m_contextManager);

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

  std::optional<pydenox::Tensor> infer(pybind11::object input,
                        // kwargs
                        std::optional<std::string> device,
                        denox::VulkanApiVersion target_env,
                        denox::DataType dtype, denox::Layout layout);

private:
  void release();

private:
  void *m_dnxBuffer;
  std::size_t m_dnxBufferSize;
  std::unordered_map<denox::RuntimeContext,
                     std::shared_ptr<denox::RuntimeModel>>
      m_runtimeModels;
  ContextManager m_contextManager; // order matters!
};

} // namespace pydenox

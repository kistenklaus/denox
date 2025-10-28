#pragma once

#include "ContextManager.hpp"
#include "Shape.hpp"
#include "Tensor.hpp"
#include "denox/common/types.hpp"
#include "denox/compiler.hpp"
#include "denox/runtime.hpp"
#include <cstddef>
#include <fmt/base.h>
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

  pybind11::object infer(pybind11::object input,
                         // kwargs
                         std::optional<std::string> device,
                         denox::VulkanApiVersion target_env,
                         denox::DataType dtype, denox::Layout layout) {

    Tensor tensor = Tensor::from(input, dtype, layout);

    denox::RuntimeContext ctx =
        m_contextManager.getContextFor(device, target_env);
    assert(ctx);

    denox::RuntimeModel model;
    if (m_runtimeModels.contains(ctx)) {
      model = *m_runtimeModels.at(ctx).get();
    } else {
      if (denox::create_runtime_model(ctx, m_dnxBuffer, m_dnxBufferSize,
                                      &model) < 0) {
        throw std::runtime_error("Failed to create runtime model.");
      }
      auto deleter = [&](denox::RuntimeModel *ptr) {
        assert(ptr != nullptr);
        fmt::println("Destroy runtime model");
        denox::destroy_runtime_model(ctx, *ptr);
        delete ptr;
      };
      auto ptr = std::shared_ptr<denox::RuntimeModel>(new denox::RuntimeModel,
                                                      deleter);
      *ptr = model;
      m_runtimeModels.insert(std::make_pair(ctx, std::move(ptr)));
    }
    assert(model);

    fmt::println("create instance");

    // denox::RuntimeInstance instance;
    // if (denox::create_runtime_instance2(ctx, model, tensor.height(),
    // tensor.width(),
    //                                 tensor.channels(), &instance) < 0) {
    //   throw std::runtime_error("Failed to create denox runtime instance.");
    // }

    // denox::eval_runtime_instance(ctx, instance, tensor.data(), void
    // **outputs)

    // denox::destroy_runtime_instance(ctx, instance);

    fmt::println("destroy instance");

    return pybind11::none();
  }

private:
  void release();

private:
  void *m_dnxBuffer;
  std::size_t m_dnxBufferSize;
  ContextManager m_contextManager;
  std::unordered_map<denox::RuntimeContext,
                     std::shared_ptr<denox::RuntimeModel>>
      m_runtimeModels;
};

} // namespace pydenox

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
      auto deleter = [ctx](denox::RuntimeModel *ptr) {
        assert(ptr != nullptr);
        denox::destroy_runtime_model(ctx, *ptr);
        delete ptr;
      };
      auto ptr = std::shared_ptr<denox::RuntimeModel>(new denox::RuntimeModel,
                                                      deleter);
      *ptr = model;
      m_runtimeModels.insert(std::make_pair(ctx, std::move(ptr)));
    }
    assert(model);

    int inputCount = denox::get_runtime_model_input_count(model);
    int outputCount = denox::get_runtime_model_output_count(model);
    if (outputCount == 0) {
      return pybind11::none();
    }
    if (inputCount != 1 || outputCount != 1) {
      throw std::runtime_error("not-implemented");
    }

    const char *inputName = denox::get_runtime_model_input_name(model, 0);
    const char *outputName = denox::get_runtime_model_output_name(model, 0);
    denox::DataType inputType =
        denox::get_runtime_model_tensor_dtype(model, inputName);
    denox::DataType outputType =
        denox::get_runtime_model_tensor_dtype(model, outputName);
    denox::Layout inputLayout =
        denox::get_runtime_model_tensor_layout(model, inputName);
    denox::Layout outputLayout =
        denox::get_runtime_model_tensor_layout(model, outputName);

    assert(inputType != denox::DataType::Auto);
    assert(outputType != denox::DataType::Auto);
    assert(inputLayout != denox::Layout::Undefined);
    assert(outputLayout != denox::Layout::Undefined);

    tensor = tensor.transform(inputType, inputLayout);

    denox::RuntimeInstance instance;
    if (denox::create_runtime_instance2(ctx, model, tensor.height(),
                                        tensor.width(), tensor.channels(),
                                        &instance) < 0) {
      throw std::runtime_error("Failed to create denox runtime instance.");
    }

    const void *inptr = tensor.data();

    std::size_t outputSize =
        denox::get_runtime_instance_tensor_byte_size(instance, outputName);

    void *outptr = std::malloc(outputSize);

    denox::eval_runtime_instance(ctx, instance, &inptr, &outptr);

    denox::destroy_runtime_instance(ctx, instance);

    return pybind11::none();
  }

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

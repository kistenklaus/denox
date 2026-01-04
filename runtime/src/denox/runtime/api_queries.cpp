#include "denox/runtime.hpp"
#include "dnx_parse_helpers.hpp"
#include "instance.hpp"
#include "model.hpp"
#include <algorithm>
#include <stdexcept>

namespace denox {

int get_runtime_model_input_count(RuntimeModel model) {
  auto m = reinterpret_cast<runtime::Model *>(model);
  return m->dnx->inputs()->size();
}

int get_runtime_model_output_count(RuntimeModel model) {
  auto m = reinterpret_cast<runtime::Model *>(model);
  return m->dnx->outputs()->size();
}

std::uint64_t get_runtime_instance_extent(RuntimeInstance instance,
                                          const char *extentName) {
  const runtime::Instance *mi =
      static_cast<const runtime::Instance *>(instance);
  const runtime::Model *m = mi->model;
  const dnx::Model *dnx = m->dnx;
  const auto [type, scalarSource] =
      dnx::getScalarSourceOfValueName(dnx, extentName);
  switch (type) {
  case dnx::ScalarSource_NONE:
    throw std::runtime_error(
        fmt::format("Extent \"{}\" does not exist.", extentName));
    return -1;
  case dnx::ScalarSource_literal:
    return dnx::parseUnsignedScalarLiteral(
        static_cast<const dnx::ScalarLiteral *>(scalarSource));
  case dnx::ScalarSource_symbolic: {
    return mi
        ->symbolValues[static_cast<const dnx::SymRef *>(scalarSource)->sid()];
  default:
    throw std::runtime_error("invalid state");
  }
  }
}

int get_runtime_instance_tensor_shape(RuntimeInstance instance,
                                      const char *tensorName, Extent *height,
                                      Extent *width, Extent *channels) {
  const runtime::Instance *mi = static_cast<runtime::Instance *>(instance);
  const auto matchesTensorName = [&](const runtime::InstanceTensorInfo &info) {
    return std::strcmp(info.name, tensorName) == 0;
  };
  const runtime::InstanceTensorInfo *info = nullptr;
  {
    const auto it =
        std::find_if(mi->inputs.begin(), mi->inputs.end(), matchesTensorName);
    if (it != mi->inputs.end()) {
      info = it.operator->();
    }
  }
  if (info == nullptr) {
    const auto it =
        std::find_if(mi->outputs.begin(), mi->outputs.end(), matchesTensorName);
    if (it != mi->outputs.end()) {
      info = it.operator->();
    }
  }
  if (info == nullptr) {
    return -1;
  }
  const std::uint32_t tensorId = info->tensor;
  const auto &tensor = mi->tensors[tensorId];

  if (height != nullptr) {
    *height = info->height;
  }
  if (width != nullptr) {
    *width = info->width;
  }
  if (channels != nullptr) {
    *channels = info->channels;
  }
  return 0;
}

std::size_t get_runtime_instance_tensor_byte_size(RuntimeInstance instance,
                                                  const char *tensorName) {

  const runtime::Instance *mi = static_cast<runtime::Instance *>(instance);
  const auto matchesTensorName = [&](const runtime::InstanceTensorInfo &info) {
    return std::strcmp(info.name, tensorName) == 0;
  };
  const runtime::InstanceTensorInfo *info = nullptr;
  {
    const auto it =
        std::find_if(mi->inputs.begin(), mi->inputs.end(), matchesTensorName);
    if (it != mi->inputs.end()) {
      info = it.operator->();
    }
  }
  if (info == nullptr) {
    const auto it =
        std::find_if(mi->outputs.begin(), mi->outputs.end(), matchesTensorName);
    if (it != mi->outputs.end()) {
      info = it.operator->();
    }
  }
  if (info == nullptr) {
    throw std::runtime_error(
        fmt::format("Tensor \"{}\" does not exist", tensorName));
  }
  return mi->tensors[info->tensor].size;
}

const char *get_runtime_model_input_name(RuntimeModel model, int index) {
  const auto *m = static_cast<runtime::Model *>(model);
  const auto *dnx = m->dnx;
  std::uint32_t inputCount = dnx->inputs()->size();
  if (index >= inputCount) {
    return nullptr;
  }
  return dnx->inputs()->Get(index)->name()->c_str();
}
const char *get_runtime_model_output_name(RuntimeModel model, int index) {
  const auto *m = static_cast<runtime::Model *>(model);
  const auto *dnx = m->dnx;
  std::uint32_t outputCount = dnx->outputs()->size();
  if (index >= outputCount) {
    return nullptr;
  }
  return dnx->outputs()->Get(index)->name()->c_str();
}

DataType get_runtime_model_tensor_dtype(RuntimeModel model,
                                        const char *tensorName) {
  const auto *m = static_cast<runtime::Model *>(model);
  const auto *dnx = m->dnx;
  const dnx::TensorInfo *info = dnx::get_tensor_info_by_name(dnx, tensorName);
  if (info == nullptr) {
    return DataType::Auto;
  }
  switch (info->type()) {
  case dnx::ScalarType_F16:
    return DataType::Float16;
  case dnx::ScalarType_F32:
    return DataType::Float32;
  case dnx::ScalarType_F64:
  case dnx::ScalarType_I16:
  case dnx::ScalarType_U16:
  case dnx::ScalarType_I32:
  case dnx::ScalarType_U32:
  case dnx::ScalarType_I64:
  case dnx::ScalarType_U64:
  default:
    throw std::runtime_error("invalid or unimplemented state.");
  }
}

Layout get_runtime_model_tensor_layout(RuntimeModel model,
                                       const char *tensorName) {
  const auto *m = static_cast<runtime::Model *>(model);
  const auto *dnx = m->dnx;
  const dnx::TensorInfo* info = dnx::get_tensor_info_by_name(dnx, tensorName);
  if (info == nullptr) {
    return Layout::Undefined;
  }
  switch (info->layout()) {
  case dnx::TensorLayout_HWC:
    return Layout::HWC;
  case dnx::TensorLayout_CHW:
    return Layout::CHW;
  case dnx::TensorLayout_CHWC8:
    return Layout::CHWC8;
  default:
    throw std::runtime_error("unreachable");
  }
}

} // namespace denox

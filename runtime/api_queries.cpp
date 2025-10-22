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

int get_runtime_instance_extent(RuntimeInstance instance,
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
                                      const char *tensorName,
                                      std::uint32_t *height,
                                      std::uint32_t *width,
                                      std::uint32_t *channels) {
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

} // namespace denox

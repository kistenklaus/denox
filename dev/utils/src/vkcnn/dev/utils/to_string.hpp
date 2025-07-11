#pragma once

#include "vkcnn/common/tensor/ActivationLayout.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
#include <string>
namespace vkcnn {

std::string_view to_string(vkcnn::FloatType type);

std::string_view to_string(vkcnn::ActivationLayout layout);
}

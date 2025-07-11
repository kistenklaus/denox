#include "./to_string.hpp"

std::string_view vkcnn::to_string(vkcnn::FloatType type) {
  if (type == FloatType::F16) {
    return "f16";
  } else if (type == FloatType::F32) {
    return "f32";
  } else if (type == FloatType::F64) {
    return "f64";
  } else {
    return "f??";
  }
}

std::string_view vkcnn::to_string(vkcnn::ActivationLayout layout) {
  if (layout == vkcnn::ActivationLayout::CHW) {
    return "CHW";
  } else if (layout == vkcnn::ActivationLayout::HWC) {
    return "HWC";
  } else if (layout == vkcnn::ActivationLayout::CHWC8) {
    return "CHWC8";
  } else if (layout == vkcnn::ActivationLayout::CHWC16) {
    return "CHWC16";
  } else {
    return "???";
  }
}

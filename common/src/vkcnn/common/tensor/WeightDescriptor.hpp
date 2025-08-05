#pragma once

#include "vkcnn/common/tensor/BiasDescriptor.hpp"
#include "vkcnn/common/tensor/FilterLayout.hpp"
#include "vkcnn/common/tensor/FilterShape.hpp"
#include "vkcnn/common/tensor/FitlerDescriptor.hpp"
#include "vkcnn/common/tensor/FloatType.hpp"
namespace vkcnn {

struct WeightDescriptor {
  struct Bias {
    BiasLayout layout;
    FloatType type;

    constexpr auto operator<=>(const Bias &) const = default;
  };

  FilterShape filterShape;
  FilterLayout filterLayout;
  FloatType filterType;
  std::optional<Bias> bias; // nullopt => no-bias.

  FilterDescriptor filterDescriptor() const {
    return FilterDescriptor(filterShape, filterLayout, filterType);
  }

  std::optional<BiasDescriptor> biasDescriptor() const {
    if (bias.has_value()) {
      return BiasDescriptor(filterShape.k, bias->layout, bias->type);
    } else {
      return std::nullopt;
    }
  }

  constexpr auto operator<=>(const WeightDescriptor &) const = default;
};

} // namespace vkcnn

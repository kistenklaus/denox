#pragma once

#include "pyvk/host/LayerDescription.hpp"
#include <memory>
#include <vector>
namespace pyvk {

class DispatchOp {
public:
  explicit DispatchOp(std::span<std::shared_ptr<LayerDescription>> layers)
      : m_layers(layers.begin(), layers.end()) {}

private:
  std::vector<std::shared_ptr<LayerDescription>> m_layers;
};

} // namespace pyvk

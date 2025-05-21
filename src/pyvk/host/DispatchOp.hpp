#pragma once

#include "pyvk/host/LayerDescription.hpp"
#include <cwchar>
#include <memory>
#include <print>
#include <vector>
namespace pyvk {

class DispatchOp {
public:
  explicit DispatchOp(std::span<std::shared_ptr<LayerDescription>> layers)
      : m_layers(layers.begin(), layers.end()) {
    assert(!m_layers.empty());
  }

  void logPretty() const {
    std::string name;
    bool first = true;
    for (const auto &layer : m_layers) {
      if (!first) {
        name += " + ";
      }
      name += layer->name;
      first = false;
    }
    std::println("- {}", name);
  }

  std::span<const std::shared_ptr<LayerDescription>> layers() const {
    return m_layers;
  }

private:
  std::vector<std::shared_ptr<LayerDescription>> m_layers;
};

} // namespace pyvk

#pragma once

#include "src/host/CNNLayerDescription.hpp"
#include <vector>
namespace pyvk {

struct CNNDescription {
  std::vector<CNNLayerDescription> layers;
};

} // namespace pyvk

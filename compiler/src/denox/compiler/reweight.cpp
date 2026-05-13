#include "denox/compiler/reweight.hpp"
#include "denox/spirv/SpirvTools.hpp"

namespace denox {

memory::vector<std::byte>
reweight(memory::span<const std::byte> dnx,
                memory::span<const std::byte> onnx,
                const diag::Logger &logger) {
  diag::Progress progress{};

}

}

#pragma once

#include "denox/compiler/Options.hpp"
#include "denox/memory/container/span.hpp"
#include "denox/compiler/frontend/model/Model.hpp"

namespace denox::onnx {

compiler::Model read(memory::span<const std::byte> raw, const compiler::Options& options);

}

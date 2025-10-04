#pragma once

#include "Options.hpp"
#include "io/fs/Path.hpp"
#include "memory/container/span.hpp"
#include "model/Model.hpp"

namespace denox::onnx {

compiler::Model read(memory::span<const std::byte> raw, io::Path onnx_dir, const compiler::Options& options);

}

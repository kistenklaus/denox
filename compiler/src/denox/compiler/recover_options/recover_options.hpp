#pragma once

#include "denox/compiler/Options.hpp"
namespace denox::compiler {

/// NOTE: Doesn't recover the exact original options that where uses to compile
/// the dnx artefact, but recovers all things, required for reweight!
std::pair<CompileOptions, std::string>
recover_options(memory::span<const std::byte> dnx);

} // namespace denox::compiler

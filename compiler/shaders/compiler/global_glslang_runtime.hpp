#pragma once

namespace denox::compiler::global_glslang_runtime {

void ensure_initialized();
void finalize();

void assume_externally_managed();

} // namespace denox::compiler::global_glslang_runtime

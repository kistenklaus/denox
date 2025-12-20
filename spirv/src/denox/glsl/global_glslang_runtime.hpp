#pragma once

namespace denox::glslang {

void ensure_initialized();
void finalize();
void assume_externally_managed();

} // namespace denox::compiler::global_glslang_runtime

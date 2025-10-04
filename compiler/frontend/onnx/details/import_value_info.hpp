#pragma once

#include "Options.hpp"
#include "frontend/onnx/details/ImportState.hpp"
#include <fmt/format.h>

namespace onnx {
class ValueInfoProto;
}

namespace denox::onnx::details {

enum ValueInfoImportContext {
  Input,
  Output,
  Hint,
};

void import_value_info(ImportState &state,
                       const ::onnx::ValueInfoProto &valueInfo,
                       ValueInfoImportContext context,
                       const compiler::Options& options);

} // namespace denox::onnx::details

#pragma once

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
                       ValueInfoImportContext context);

} // namespace denox::onnx::details

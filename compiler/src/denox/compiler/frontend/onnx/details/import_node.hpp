#pragma once

#include "denox/compiler/frontend/onnx/details/ImportState.hpp"
namespace onnx {
class NodeProto;
}

namespace denox::onnx::details {

void import_node(ImportState &state, const ::onnx::NodeProto &node);
}

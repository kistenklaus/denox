#pragma once

#include "vkcnn/host/ops/OpConv2d.hpp"
namespace vkcnn::codegen {

void direct_conv2d(const OpConv2d& op, std::string& source);

}

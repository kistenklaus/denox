#pragma once

#include "device_info/ApiVersion.hpp"
#include <vulkan/vulkan.hpp>

namespace denox::compiler::device_info::query {

vk::Instance create_query_instance(ApiVersion);
}

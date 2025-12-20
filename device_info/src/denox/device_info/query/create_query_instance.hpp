#pragma once

#include "denox/device_info/ApiVersion.hpp"
#include <vulkan/vulkan.hpp>

namespace denox {

vk::Instance create_query_instance(ApiVersion&);
}

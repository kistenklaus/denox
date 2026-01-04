#pragma once

#include "dnx.h"
#include <cstdint>
namespace denox::dnx {

std::uint64_t parseUnsignedScalarLiteral(const dnx::ScalarLiteral *literal);

std::pair<dnx::ScalarSource, const void *>
getScalarSourceOfValueName(const dnx::Model *dnx, const char *valueName);

std::uint64_t
parseUnsignedScalarSource(dnx::ScalarSource scalar_source_type,
                          const void *scalar_source,
                          std::span<const std::int64_t> symbolValues);
void memcpyPushConstantField(void *dst, const dnx::PushConstantField *field,
                             std::span<const std::int64_t> symbolValues);

const char *reverse_value_name_search(const dnx::Model *dnx,
                                      const dnx::ScalarSource source_type,
                                      const void *source);

const dnx::TensorInfo *get_tensor_info_by_name(const dnx::Model *dnx,
                                               const char *name);

} // namespace denox::dnx

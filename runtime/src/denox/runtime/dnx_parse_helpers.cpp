#include "dnx_parse_helpers.hpp"
#include <stdexcept>

namespace denox::dnx {

std::uint64_t parseUnsignedScalarLiteral(const dnx::ScalarLiteral *literal) {
  switch (literal->dtype()) {
  case dnx::ScalarType_I16: {
    std::int16_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::int16_t));
    return static_cast<std::size_t>(x);
  }
  case dnx::ScalarType_U16: {
    std::uint16_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::uint16_t));
    return static_cast<std::size_t>(x);
  }
  case dnx::ScalarType_I32: {
    std::int32_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::int32_t));
    return static_cast<std::size_t>(x);
    break;
  }
  case dnx::ScalarType_U32: {
    std::uint32_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::uint32_t));
    return static_cast<std::size_t>(x);
  }
  case dnx::ScalarType_I64: {
    std::int64_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::int64_t));
    return static_cast<std::size_t>(x);
  }
  case dnx::ScalarType_U64: {
    std::uint64_t x;
    std::memcpy(&x, literal->bytes()->data(), sizeof(std::uint64_t));
    return static_cast<std::size_t>(x);
  }
  case dnx::ScalarType_F16:
  case dnx::ScalarType_F32:
  case dnx::ScalarType_F64:
    throw std::runtime_error("Invalid size dtype.");
  default:
    throw std::runtime_error("Unexpected size dtype.");
  }
}

std::pair<dnx::ScalarSource, const void *>
getScalarSourceOfValueName(const dnx::Model *dnx, const char *valueName) {
  for (std::size_t i = 0; i < dnx->value_names()->size(); ++i) {
    const dnx::ValueName *v = dnx->value_names()->Get(static_cast<unsigned int>(i));
    if (std::strcmp(v->name()->c_str(), valueName) == 0) {
      return std::make_pair(v->value_type(), v->value());
    }
  }
  return std::make_pair(dnx::ScalarSource_NONE, nullptr);
}

std::uint64_t
parseUnsignedScalarSource(dnx::ScalarSource scalar_source_type,
                          const void *scalar_source,
                          std::span<const std::int64_t> symbolValues) {
  switch (scalar_source_type) {
  case ScalarSource_literal:
    return parseUnsignedScalarLiteral(
        static_cast<const dnx::ScalarLiteral *>(scalar_source));
  case ScalarSource_symbolic:
    return static_cast<uint64_t>(symbolValues[static_cast<const dnx::SymRef *>(scalar_source)->sid()]);
  default:
  case ScalarSource_NONE:
    throw std::runtime_error("invalid scalar source");
  }
}

void memcpyPushConstantField(void *dst, const dnx::PushConstantField *field,
                             std::span<const std::int64_t> symbolValues) {
  std::byte *pDst = static_cast<std::byte *>(dst) + field->offset();
  switch (field->dtype()) {
  case dnx::ScalarType_I16:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::int16_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::int16_t v = static_cast<std::int16_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::int16_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_U16:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::uint16_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::uint16_t v = static_cast<std::uint16_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::uint16_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_I32:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::int32_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::int32_t v = static_cast<std::int32_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::int32_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_U32:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::uint32_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::int32_t v = static_cast<std::int32_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::uint32_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_I64:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::int64_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::int64_t v = static_cast<std::int64_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::int64_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_U64:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::uint64_t));
      break;
    case dnx::ScalarSource_symbolic: {
      std::uint64_t v = static_cast<std::uint64_t>(
          symbolValues[field->source_as_symbolic()->sid()]);
      std::memcpy(pDst, &v, sizeof(std::uint64_t));
      break;
    }
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_F16:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(std::uint16_t));
      break;
    case dnx::ScalarSource_symbolic:
    case dnx::ScalarSource_NONE:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  case dnx::ScalarType_F32:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(float));
      break;
    case dnx::ScalarSource_NONE:
    case dnx::ScalarSource_symbolic:
      throw std::runtime_error("invalid scalar source");
      break;
    }
    break;
  case dnx::ScalarType_F64:
    switch (field->source_type()) {
    case dnx::ScalarSource_literal:
      std::memcpy(pDst, field->source_as_literal()->bytes()->data(),
                  sizeof(double));
      break;
    case dnx::ScalarSource_NONE:
    case dnx::ScalarSource_symbolic:
      throw std::runtime_error("invalid scalar source");
    }
    break;
  }
}

const char* reverse_value_name_search(const dnx::Model* dnx, const dnx::ScalarSource source_type,
    const void* source) {
  std::size_t count = dnx->value_names()->size();
  for (std::size_t v = 0; v < count; ++v) {
    const dnx::ValueName* valueName = dnx->value_names()->Get(static_cast<unsigned int>(v));
    if (valueName->value_type() != source_type) {
      continue;
    }
    if (valueName->value_type() == dnx::ScalarSource_symbolic) {
      const dnx::SymRef* valueNameSymRef = valueName->value_as_symbolic();
      const dnx::SymRef* sourceRef = static_cast<const dnx::SymRef*>(source);
      if (sourceRef->sid() == valueNameSymRef->sid()) {
        return valueName->name()->c_str();
      }
    } else if(valueName->value_type() == dnx::ScalarSource_literal) {
      const dnx::ScalarLiteral* valueLiteral =  valueName->value_as_literal();
      const dnx::ScalarLiteral* sourceLiteral = static_cast<const dnx::ScalarLiteral*>(source);
      if (valueLiteral->dtype() != sourceLiteral->dtype()) {
        continue;
      }
      assert(valueLiteral->bytes()->size() == sourceLiteral->bytes()->size());
      if (std::memcmp(valueLiteral->bytes()->data(), sourceLiteral->bytes()->data(), valueLiteral->bytes()->size()) == 0) {
        return valueName->name()->c_str();
      }
    }
  }
  return nullptr;
}

const dnx::TensorInfo* get_tensor_info_by_name(const dnx::Model* dnx, 
    const char* name) {
  std::size_t inputCount = dnx->inputs()->size();
  for (std::size_t i = 0; i< inputCount; ++i) {
    uint32_t input = dnx->inputs()->Get(static_cast<unsigned int>(i));
    auto info =  dnx->tensors()->Get(input)->info();
    if (info == nullptr) {
      continue;
    }
    if (std::strcmp(info->name()->c_str(), name) == 0) {
      return info;
    }
  }
  std::size_t outputCount = dnx->outputs()->size();
  for (std::size_t o = 0; o < outputCount; ++o)  {
    uint32_t output = dnx->outputs()->Get(static_cast<unsigned int>(o));
    auto info = dnx->tensors()->Get(output)->info();
    if (info == nullptr) {
      continue;
    }
    if (std::strcmp(info->name()->c_str(), name) == 0) {
      return info;
    }
  }
  return nullptr;
}


} // namespace denox::dnx

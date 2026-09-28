#pragma once
#include "denox/cli/io/InputStream.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"
#include <cstring>

class PngInputStream {
public:
  PngInputStream(InputStream *stream) : m_stream(stream) {}

  denox::memory::optional<denox::memory::ActivationTensor>
  read_image();

  static bool is_png(uint64_t magic) {
    constexpr unsigned char signature[8]{0x89, 'P',  'N',  'G',
                                         0x0D, 0x0A, 0x1A, 0x0A};
    return std::memcmp(&magic, signature, sizeof(signature)) == 0;
  }

private:
  InputStream *m_stream;
};

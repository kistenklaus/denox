#pragma once

#include "denox/cli/io/InputStream.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/dtype/dtype.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"
#include <cstring>

class NpyInputStream {
public:
  NpyInputStream(InputStream *stream) : m_stream(stream) {}

  denox::memory::optional<denox::memory::ActivationTensor>
  read_tensor();


  static bool is_npy(uint64_t magic) {
    constexpr char signature[] = "\x93NUMPY";
    return std::memcmp(&magic, signature, 6) == 0;
  }

private:
  InputStream *m_stream;
};

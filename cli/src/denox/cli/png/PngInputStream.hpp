#pragma once
#include "denox/cli/io/InputStream.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"

class PngInputStream {
public:
  PngInputStream(InputStream* stream) : m_stream(stream) {}

  denox::memory::optional<denox::memory::ActivationTensor> read_image();

private:
  InputStream* m_stream;
};

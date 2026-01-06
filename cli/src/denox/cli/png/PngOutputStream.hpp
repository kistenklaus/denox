#pragma once
#include "denox/cli/io/OutputStream.hpp"
#include "denox/memory/container/optional.hpp"
#include "denox/memory/tensor/ActivationTensor.hpp"

class PngOutputStream {
public:
  PngOutputStream(OutputStream* stream) : m_stream(stream) {}

  void write_image(const denox::memory::ActivationTensor &image);

private:
  OutputStream* m_stream;
};

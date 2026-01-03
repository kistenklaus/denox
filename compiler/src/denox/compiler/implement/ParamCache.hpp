#pragma once

#include "denox/memory/container/hashmap.hpp"
#include "denox/memory/container/small_vector.hpp"
#include "denox/memory/container/vector.hpp"
#include "denox/memory/tensor/BiasDescriptor.hpp"
#include "denox/memory/tensor/BiasTensor.hpp"
#include "denox/memory/tensor/FilterTensor.hpp"
#include <fmt/ostream.h>
#include <memory>
namespace denox::compiler {

class ParamCache {
public:
  std::shared_ptr<memory::vector<std::byte>>
  convert(memory::BiasDescriptor descriptor, memory::BiasTensorConstView view) {

    const void *ptr = static_cast<const void *>(view.data());
    auto& cache = m_biasCache[ptr];

    auto it = std::ranges::find_if(
        cache.lines, [&](const auto &line) { return line.tag == descriptor; });
    if (it != cache.lines.end()) {
      return it->data;
    }
    // does not exist in cache yet.
    memory::BiasTensor tensor{
        descriptor,
        view,
    };
    auto data = std::make_shared<memory::vector<std::byte>>(
        tensor.span().begin(), tensor.span().end());
    cache.lines.push_back(BiasCacheLine{
        .tag = descriptor,
        .data = data,
    });
    return data;
  }

  std::shared_ptr<memory::vector<std::byte>>
  convert(memory::FilterDescriptor descriptor,
          memory::FilterTensorConstView view) {
    const void *ptr = static_cast<const void *>(view.data());
    auto& cache = m_filterCache[ptr];

    auto it = std::ranges::find_if(
        cache.lines, [&](const auto &line) { return line.tag == descriptor; });
    if (it != cache.lines.end()) {
      return it->data;
    }

    memory::FilterTensor tensor{
        descriptor,
        view,
    };

    auto data = std::make_shared<memory::vector<std::byte>>(
        tensor.span().begin(), tensor.span().end());
    cache.lines.push_back(FilterCacheLine{
        .tag = descriptor,
        .data = data,
    });
    return data;
  }

private:
  struct BiasCacheLine {
    memory::BiasDescriptor tag;
    std::shared_ptr<memory::vector<std::byte>> data;
  };

  struct BiasCache {
    memory::small_vector<BiasCacheLine, 4> lines;
  };

  struct FilterCacheLine {
    memory::FilterDescriptor tag;
    std::shared_ptr<memory::vector<std::byte>> data;
  };

  struct FilterCache {
    memory::small_vector<FilterCacheLine, 10> lines;
  };

  memory::hash_map<const void *, BiasCache> m_biasCache;
  memory::hash_map<const void *, FilterCache> m_filterCache;
};

} // namespace denox::compiler

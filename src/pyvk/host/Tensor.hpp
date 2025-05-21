#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <span>
#include <cassert>
#include <vector>

namespace pyvk {

template <std::size_t DIM> using TensorShape = std::array<std::size_t, DIM>;
template <std::size_t DIM> using TensorIndex = std::array<std::size_t, DIM>;

template <typename T>
concept TensorFormat =
    requires {
      { T::rank } -> std::convertible_to<std::size_t>;
    } && requires(const TensorShape<T::rank> &shape,
                  const TensorIndex<T::rank> &index) {
      { T::offset(shape, index) } -> std::convertible_to<std::size_t>;
    };

struct TensorFormat_OIHW {
public:
  static constexpr std::size_t rank = 4;
  static std::size_t offset(const TensorShape<rank> &shape,
                            const TensorIndex<rank> &index) {
    const auto &[O, I, H, W] = shape;
    const auto &[o, i, h, w] = index;
    return ((o * I + i) * H + h) * W + w;
  }
};

template <typename T, TensorFormat Format> class Tensor {
public:
  using format = Format;
  static constexpr std::size_t rank = format::rank;
  using shape_type = TensorShape<rank>;
  using index_type = TensorIndex<rank>;
  explicit Tensor(const shape_type &shape, const T &v = 0)
      : m_shape(shape), m_tensor(std::accumulate(shape.begin(), shape.end(), 1,
                                                 std::multiplies{}),
                                 v) {}

  explicit Tensor(const shape_type &shape, std::span<T> values)
      : m_shape(shape), m_tensor(values.begin(), values.end()) {
    auto n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{});
    assert(n == m_tensor.size());
  }

  explicit Tensor(const shape_type &shape, std::vector<T> values)
      : m_shape(shape), m_tensor(std::move(values)) {
    auto n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies{});
    assert(n == m_tensor.size());
  }

  template <typename NewT, TensorFormat NewFormat> Tensor transform() {
    static_assert(rank == NewFormat::rank,
                  "Rank mismatch in tensor transform.");

    Tensor<NewT, NewFormat> result(m_shape);
    TensorIndex<rank> index = {};

    do {
      std::size_t src_offset = format::offset(m_shape, index);
      std::size_t dst_offset = NewFormat::offset(m_shape, index);
      result.raw()[dst_offset] = static_cast<NewT>(m_tensor[src_offset]);
    } while (next_index<rank>(m_shape, index));

    return result;
  }

  std::size_t size() const { return m_tensor.size(); }

  const shape_type &shape() const { return m_shape; }

  T &operator[](const index_type &index) {
    std::size_t idx = format::offset(m_shape, index);
    return m_tensor[idx];
  }

  const T *data() const { return m_tensor.data(); }
  T *data() { return m_tensor.data(); }

private:
  template <std::size_t N>
  bool next_index(const TensorShape<N> &shape, TensorIndex<N> &index) {
    for (std::size_t i = N; i-- > 0;) {
      if (++index[i] < shape[i])
        return true;
      index[i] = 0;
    }
    return false; // reached end
  }

  shape_type m_shape;
  std::vector<T> m_tensor;
};

} // namespace pyvk

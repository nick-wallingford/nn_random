#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <ostream>

namespace nnla {

template <typename T, size_t Rows, size_t Columns> class alignas(64) matrix;

template <typename T, size_t N> class alignas(64) vector {
  std::array<T, N> d;

public:
  using is_vector_view = std::true_type;

  vector(std::nullopt_t) noexcept {}
  constexpr vector() noexcept : d{} {}
  constexpr vector(const std::array<T, N> &i) noexcept : d{i} {}
  constexpr vector(const vector &) noexcept = default;
  constexpr vector &operator=(const vector &) noexcept = default;

  template <typename Self> [[nodiscard]] constexpr auto &operator[](this Self &&self, size_t index) {
    assert(index < N);
    return self.d[index];
  }

  static constexpr size_t size = N;
  template <typename Self> [[nodiscard]] constexpr auto begin(this Self &&self) { return self.d.begin(); }
  template <typename Self> [[nodiscard]] constexpr auto end(this Self &&self) { return self.d.end(); }

  [[nodiscard]] constexpr bool operator==(const vector &o) const noexcept {
    for (size_t i = 0; i < N; i++)
      if (d[i] != o[i])
        return false;
    return true;
  }

  [[nodiscard]] constexpr vector operator-(const vector &o) const noexcept {
    vector r;
    for (size_t i = 0; i < N; i++)
      r[i] = d[i] - o[i];
    return r;
  }

  [[nodiscard]] constexpr vector operator+(const vector &o) const noexcept {
    vector r;
    for (size_t i = 0; i < N; i++)
      r[i] = d[i] + o[i];
    return r;
  }

  [[nodiscard]] constexpr vector operator*(const vector &o) const noexcept {
    vector r;
    for (size_t i = 0; i < N; i++)
      r[i] = d[i] * o[i];
    return r;
  }

  constexpr vector &operator*=(T o) noexcept {
    for (T &x : d)
      x *= o;
    return *this;
  }

  constexpr vector &operator/=(T o) noexcept { return *this *= 1 / o; }

  constexpr vector &operator*=(const vector &o) noexcept {
    for (size_t i = 0; i < N; i++)
      d[i] *= o[i];
    return *this;
  }

  constexpr vector &operator+=(const vector &o) noexcept {
    for (size_t i = 0; i < N; i++)
      d[i] += o.d[i];
    return *this;
  }

  constexpr vector &operator-=(const vector &o) noexcept {
    for (size_t i = 0; i < N; i++)
      d[i] -= o.d[i];
    return *this;
  }

  [[nodiscard]] constexpr vector operator*(T x) const noexcept {
    vector r = *this;
    for (T &y : r.d)
      y *= x;
    return r;
  }

  [[nodiscard]] constexpr T err() const noexcept {
    T r = 0;
    for (const T x : d)
      r += x * x;
    return r;
  }

  [[nodiscard]] constexpr T sum() const noexcept {
    T r = 0;
    const T *a = d.data();
    for (size_t i = N; i--;)
      r += *a++;
    return r;
  }

  template <size_t Columns>
  [[nodiscard]] constexpr vector<T, Columns> operator*(const matrix<T, N, Columns> &) const noexcept;

  template <size_t Columns>
  [[nodiscard]] constexpr matrix<T, N, Columns> outer_product(const vector<T, Columns> &) const noexcept;
};

} // namespace nnla

#include "matrix.hpp"

namespace nnla {

template <typename T, size_t N>
template <size_t Columns>
[[nodiscard]] constexpr matrix<T, N, Columns> vector<T, N>::outer_product(const vector<T, Columns> &b) const noexcept {
  return matrix<T, N, 1>{d} * matrix<T, 1, Columns>{b.d};
}

template <typename T, size_t N>
template <size_t Columns>
[[nodiscard]] constexpr vector<T, Columns> vector<T, N>::operator*(const matrix<T, N, Columns> &m) const noexcept {
  vector<T, Columns> r{};
  auto a = begin();
  auto b = m.begin();
  for (size_t i = N; i--; ++a) {
    auto y = r.begin();
    for (size_t j = Columns; j--;)
      *y++ += *a * *b++;
  }
  return r;
}

/*
  static_assert(vector<int, 3>{(matrix<int, 1, 3>{{2, 3, 4}} * matrix<int, 3, 3>{{5, 6, 7, 8, 9, 10, 11, 12, 13}}).d} ==
            vector<int, 3>{{2, 3, 4}} * matrix<int, 3, 3>{{5, 6, 7, 8, 9, 10, 11, 12, 13}});
*/
} // namespace nnla

template <typename T, size_t N> std::ostream &operator<<(std::ostream &o, const nnla::vector<T, N> &v) {
  o << '[';
  for (size_t i = 0; i < N; i++)
    o << std::format("{:10} ", v[i]);
  o << ']';
  return o;
}

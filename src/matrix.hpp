#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <optional>
#include <ostream>
#include <random>

namespace nnla {

template <typename T, size_t N> class alignas(64) vector;

template <typename T, size_t Rows, size_t Columns> class alignas(64) matrix {
  std::array<T, Columns * Rows> d;

  [[nodiscard]] __attribute__((noinline)) T similarity(const T *const A, size_t n) const noexcept {
    T a = 0;
    T b = 0;
    T ab = 0;
    const T *const B = (*this)[n];
    for (size_t i = 0; i < Columns; i++) {
      a += A[i] * A[i];
      b += B[i] * B[i];
      ab += A[i] * B[i];
    }

    ab /= std::sqrt(a);
    ab /= std::sqrt(b);
    return ab;
  }

  [[nodiscard]] constexpr T similarity(size_t m, size_t n) const noexcept { return similarity((*this)[m], n); }

public:
  matrix(std::nullopt_t) noexcept {}
  constexpr matrix() noexcept : d{} {}
  constexpr matrix(const std::array<T, Columns * Rows> &i) noexcept : d{i} {}
  constexpr matrix(const matrix &) noexcept = default;
  constexpr matrix &operator=(const matrix &) noexcept = default;

  [[nodiscard]] constexpr const T *operator[](size_t y) const noexcept { return d.data() + y * Columns; }
  [[nodiscard]] constexpr T *operator[](size_t y) noexcept { return d.data() + y * Columns; }

  static constexpr size_t size = Rows * Columns;
  template <typename Self> [[nodiscard]] constexpr auto begin(this Self &&self) { return self.d.begin(); }
  template <typename Self> [[nodiscard]] constexpr auto end(this Self &&self) { return self.d.end(); }

  template <size_t O>
  [[nodiscard]] constexpr matrix<T, Rows, O> operator*(const matrix<T, Columns, O> &o) const noexcept {
    matrix<T, Rows, O> r;
    auto in_l = begin();
    auto out = r.begin();
    for (size_t y = 0; y < Rows; y++) {
      auto b = o.begin();
      for (size_t x = 0; x < Columns; x++) {
        const T c = *in_l++;
        for (size_t z = 0; z < O; z++)
          out[z] += c * *b++;
      }
      out += O;
    }

    return r;
  };

  constexpr matrix &operator*=(const T x) noexcept {
    for (size_t i = 0; i < Rows * Columns; i++)
      d[i] *= x;
    return *this;
  }

  [[nodiscard]] constexpr matrix operator-(const matrix &o) const noexcept {
    matrix r;
    for (size_t i = 0; i < Rows * Columns; i++)
      r[i] = d[i] - o.d[i];
    return r;
  }

  [[nodiscard]] constexpr matrix operator+(const matrix &o) const noexcept {
    matrix r;
    for (size_t i = 0; i < Rows * Columns; i++)
      r[i] = d[i] + o.d[i];
    return r;
  }

  constexpr matrix &operator+=(const matrix &o) noexcept {
    for (size_t i = 0; i < Rows * Columns; i++)
      d[i] += o.d[i];
    return *this;
  }

  constexpr void add_outer_product(const vector<T, Rows> &, const vector<T, Columns> &) noexcept;

  [[nodiscard]] constexpr vector<T, Rows> operator*(const vector<T, Columns> &) const noexcept;

  [[nodiscard]] constexpr bool operator==(const matrix &o) const noexcept {
    for (size_t i = 0; i < Rows * Columns; i++)
      if (d[i] != o.d[i])
        return false;
    return true;
  }

  template <typename Rand> void reduce_similarity(Rand &random, std::normal_distribution<T> &d) noexcept;
};

static_assert(matrix<int, 2, 3>{{1, 2, 3, 4, 5, 6}} *
                  matrix<int, 3, 4>{{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}} ==
              matrix<int, 2, 4>{{74, 80, 86, 92, 173, 188, 203, 218}});

} // namespace nnla

template <typename T, size_t Rows, size_t Columns>
std::ostream &operator<<(std::ostream &o, const nnla::matrix<T, Rows, Columns> &m) {
  for (size_t i = 0; i < Rows; i++) {
    o << '[';
    for (size_t j = 0; j < Columns; j++)
      o << std::format("{:10} ", m[i][j]);
    o << "]\n";
  }
  return o;
}

#include "vector.hpp"

namespace nnla {

template <typename T, size_t Rows, size_t Columns>
[[nodiscard]] constexpr vector<T, Rows>
matrix<T, Rows, Columns>::operator*(const vector<T, Columns> &v) const noexcept {
  vector<T, Rows> r{};
  const T *a = d.data();
  auto c = r.begin();
  for (size_t i = Rows; i--;) {
    auto b = v.begin();
    for (size_t j = Columns; j--;)
      *c += *a++ * *b++;
    ++c;
  }
  return r;
}

template <typename T, size_t Rows, size_t Columns>
template <typename Rand>
void matrix<T, Rows, Columns>::reduce_similarity(Rand &random, std::normal_distribution<T> &d) noexcept {
  static constexpr T max_sim = static_cast<T>(31) / static_cast<T>(32);
  for (size_t i = 0; i < Rows; i++)
    for (size_t j = 0; j < i; j++)
      if (const T sim = similarity(i, j); std::abs(sim) > max_sim) {
        std::cout << "found similarity:\t" << j << '\t' << i << '\t' << sim << std::endl;
        for (size_t k = 0; k < Columns; k++)
          (*this)[j][k] = d(random);
      }
}

template <typename T, size_t Rows, size_t Columns>
constexpr void matrix<T, Rows, Columns>::add_outer_product(const vector<T, Rows> &a_,
                                                           const vector<T, Columns> &b_) noexcept {
  auto a = a_.begin();
  T *o = d.data();
  for (size_t i = Rows; i--;) {
    auto b = b_.begin();
    for (size_t j = Columns; j--;)
      *o++ += *a * *b++;
    ++a;
  }
}

static_assert(matrix<int, 2, 3>{{1, 2, 3, 4, 5, 6}} * vector<int, 3>{{7, 8, 9}} == vector<int, 2>{{50, 122}});

} // namespace nnla

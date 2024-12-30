#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <ostream>

template <typename T, size_t N> struct alignas(64) vector;

template <typename T, size_t Rows, size_t Columns> struct alignas(64) matrix {
  std::array<T, Columns * Rows> d;

  matrix(std::nullopt_t) noexcept {}
  constexpr matrix() noexcept : d{} {}
  constexpr matrix(const std::array<T, Columns * Rows> &i) noexcept : d{i} {}
  constexpr matrix(const matrix &) noexcept = default;
  constexpr matrix &operator=(const matrix &) noexcept = default;

  [[nodiscard]] constexpr const T *operator[](size_t y) const noexcept { return d.data() + y * Columns; }
  [[nodiscard]] constexpr T *operator[](size_t y) noexcept { return d.data() + y * Columns; }

  template <size_t O>
  [[nodiscard]] constexpr matrix<T, Rows, O> operator*(const matrix<T, Columns, O> &o) const noexcept {
    matrix<T, Rows, O> r;
    const T *in_l = d.data();
    T *out = r.d.data();
    for (size_t y = 0; y < Rows; y++) {
      const T *b = o[0];
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
      r[i] = d[i] - o[i];
    return r;
  }

  [[nodiscard]] constexpr matrix operator+(const matrix &o) const noexcept {
    matrix r;
    for (size_t i = 0; i < Rows * Columns; i++)
      r[i] = d[i] + o[i];
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
};

static_assert(matrix<int, 2, 3>{{1, 2, 3, 4, 5, 6}} *
                  matrix<int, 3, 4>{{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}} ==
              matrix<int, 2, 4>{{74, 80, 86, 92, 173, 188, 203, 218}});

#include "vector.hpp"

template <typename T, size_t Rows, size_t Columns>
std::ostream &operator<<(std::ostream &o, const matrix<T, Rows, Columns> &m) {
  for (size_t i = 0; i < Rows; i++) {
    o << '[';
    for (size_t j = 0; j < Columns; j++)
      o << std::format("{:10} ", m[i][j]);
    o << "]\n";
  }
  return o;
}

template <typename T, size_t Rows, size_t Columns>
[[nodiscard]] constexpr vector<T, Rows>
matrix<T, Rows, Columns>::operator*(const vector<T, Columns> &v) const noexcept {
  vector<T, Rows> r{};
  const T *a = d.data();
  T *c = r.d.data();
  for (size_t i = Rows; i--;) {
    const T *b = v.d.data();
    for (size_t j = Columns; j--;)
      *c += *a++ * *b++;
    c++;
  }
  return r;
}

template <typename T, size_t Rows, size_t Columns>
constexpr void matrix<T, Rows, Columns>::add_outer_product(const vector<T, Rows> &a_,
                                                           const vector<T, Columns> &b_) noexcept {
  const T *a = a_.d.data();
  T *o = d.data();
  for (size_t i = Rows; i--;) {
    const T *b = b_.d.data();
    for (size_t j = Columns; j--;)
      *o++ += *a * *b++;
    ++a;
  }
}

static_assert(matrix<int, 2, 3>{{1, 2, 3, 4, 5, 6}} * vector<int, 3>{{7, 8, 9}} == vector<int, 2>{{50, 122}});

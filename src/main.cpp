#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>

template <typename T, size_t N> struct vector;

template <typename T, size_t Rows, size_t Columns> struct matrix {
  std::array<T, Columns * Rows> d;

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

  void square_plus() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x += std::sqrt(x2);
      if constexpr (std::is_integral_v<T>)
        x >>= 1;
      else {
        static constexpr T half = static_cast<T>(1) / static_cast<T>(2);
        x *= half;
      }
    }
  }

  void d_square_plus() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x /= std::sqrt(x2);
      x += 1;
      if constexpr (std::is_integral_v<T>)
        x >>= 1;
      else {
        static constexpr T half = static_cast<T>(1) / static_cast<T>(2);
        x *= half;
      }
    }
  }

  void square_sigmoid() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x /= std::sqrt(x2);
    }
  }

  void d_square_sigmoid() noexcept {
    for (T &x : d) {
      x *= x;
      x += 1;
      [[assume(x > 0)]];
      x = std::sqrt(x) / (x * x);
    }
  }

  [[nodiscard]] constexpr vector<T, Rows> operator*(const vector<T, Columns> &) const noexcept;

  [[nodiscard]] constexpr bool operator==(const matrix &o) const noexcept {
    for (size_t i = 0; i < Rows * Columns; i++)
      if (d[i] != o.d[i])
        return false;
    return true;
  }
};

template <typename T, size_t Rows, size_t Columns>
std::ostream &operator<<(std::ostream &o, const matrix<T, Rows, Columns> &m) {
  for (size_t i = 0; i < Rows; i++) {
    std::cout << '[';
    for (size_t j = 0; j < Columns; j++)
      std::cout << std::format("{:10} ", m[i][j]);
    std::cout << "]\n";
  }
  return o;
}

static_assert(matrix<int, 2, 3>{{1, 2, 3, 4, 5, 6}} *
                  matrix<int, 3, 4>{{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}} ==
              matrix<int, 2, 4>{{74, 80, 86, 92, 173, 188, 203, 218}});

template <typename T, size_t N> struct vector {
  std::array<T, N> d;
  constexpr vector() noexcept : d{} {}
  constexpr vector(const std::array<T, N> &i) noexcept : d{i} {}
  constexpr vector(const vector &) noexcept = default;
  constexpr vector &operator=(const vector &) noexcept = default;

  [[nodiscard]] constexpr T operator[](size_t i) const noexcept { return d[i]; }
  [[nodiscard]] constexpr T &operator[](size_t i) noexcept { return d[i]; }
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

  [[nodiscard]] constexpr vector operator*(T x) const noexcept {
    vector r = *this;
    for (T &y : r.d)
      y *= x;
    return r;
  }

  void square_plus() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x += std::sqrt(x2);
      if constexpr (std::is_integral_v<T>)
        x >>= 1;
      else {
        static constexpr T half = static_cast<T>(1) / static_cast<T>(2);
        x *= half;
      }
    }
  }

  void d_square_plus() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x /= std::sqrt(x2);
      x += 1;
      if constexpr (std::is_integral_v<T>)
        x >>= 1;
      else {
        static constexpr T half = static_cast<T>(1) / static_cast<T>(2);
        x *= half;
      }
    }
  }

  void actual_sigmoid() noexcept {
    for (T &x : d)
      x = 1 / (1 + std::exp(-x));
  }

  void d_actual_sigmoid() noexcept {
    for (T &x : d) {
      const T num = std::exp(-x);
      T den = 1 + num;
      den *= den;
      x = num / den;
    }
  }

  void square_sigmoid() noexcept {
    for (T &x : d) {
      const T x2 = x * x + 1;
      [[assume(x2 > 0)]];
      x /= std::sqrt(x2);
    }
  }

  void d_square_sigmoid() noexcept {
    for (T &x : d) {
      x *= x;
      x += 1;
      [[assume(x > 0)]];
      x = std::sqrt(x) / (x * x);
    }
  }

  [[nodiscard]] constexpr T loss(const vector &o) const noexcept {
    T r;
    constexpr auto sqr = [](const T x) -> T { return x * x; };
    for (size_t i = 0; i < N; i++)
      r += (d[i] - o[i]) * (d[i] - o[i]);
    return r;
  }

  [[nodiscard]] constexpr vector err(const vector &o) const noexcept {
    vector r_;
    T *r = r_.d.data();
    const T *a = d.data();
    const T *b = o.d.data();
    for (size_t i = N; i--;) {
      const T x = *a++ - *b++;
      *r++ = x * x;
    }
    return r_;
  }

  [[nodiscard]] constexpr T sum() const noexcept {
    T r = 0;
    const T *a = d.data();
    for (size_t i = N; i--;)
      r += *a++;
    return r;
  }

  template <size_t Columns>
  [[nodiscard]] constexpr vector<T, Columns> operator*(const matrix<T, N, Columns> &m) const noexcept {
    vector<T, Columns> r{};
    const T *a = d.data();
    const T *b = m.d.data();
    for (size_t i = N; i--; ++a) {
      T *y = r.d.data();
      for (size_t j = Columns; j--;)
        *y++ += *a * *b++;
    }
    return r;
  }

  template <size_t Columns>
  [[nodiscard]] constexpr matrix<T, N, Columns> outer_product(const vector<T, Columns> &o) const noexcept {
    return matrix<T, N, 1>{d} * matrix<T, 1, Columns>{o.d};
  }
};

static_assert(vector<int, 3>{(matrix<int, 1, 3>{{2, 3, 4}} * matrix<int, 3, 3>{{5, 6, 7, 8, 9, 10, 11, 12, 13}}).d} ==
              vector<int, 3>{{2, 3, 4}} * matrix<int, 3, 3>{{5, 6, 7, 8, 9, 10, 11, 12, 13}});

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

template <typename T, size_t N> std::ostream &operator<<(std::ostream &o, const vector<T, N> &v) {
  std::cout << '[';
  for (size_t i = 0; i < N; i++)
    std::cout << std::format("{:10} ", v[i]);
  std::cout << ']';
  return o;
}

int main() {
  using matrix_t = matrix<float, 2, 2>;
  using vector_t = vector<float, 2>;

  matrix_t W{{6, -3, -2, 5}};
  matrix_t V{{1, -2, .25f, 2}};
  const std::array<vector_t, 2> X{vector_t{{3, 1}}, vector_t{{-1, 4}}};
  const std::array<vector_t, 2> T{vector_t{{1, 0}}, vector_t{{0, 1}}};

  float alpha = -1;

  for (int iteration = 0; std::isnormal(alpha); iteration++) {
    matrix_t grad_V, grad_W;
    for (size_t i = 0; i < 2; i++) {
      const vector_t H_in = X[i] * W;
      vector_t H = H_in;
      H.actual_sigmoid();

      const vector_t Y_in = H * V;
      vector_t Y = Y_in;
      Y.actual_sigmoid();

      vector_t E = Y - T[i];
      // vector_t E_sqr = E * E;
      if (!(iteration & (iteration - 1)))
        std::cout << (E * E).sum() << std::endl;

      vector_t grad_Y = E + E;
      vector_t grad_Y_in = Y * (vector_t{{1, 1}} - Y) * grad_Y;

      grad_V.add_outer_product(H, grad_Y_in);

      vector_t grad_H = V * grad_Y_in;
      vector_t grad_H_in = H * (vector_t{{1, 1}} - H) * grad_H;
      grad_W.add_outer_product(X[i], grad_H_in);
    }

    grad_W *= alpha;
    grad_V *= alpha;
    alpha *= 1 - std::numeric_limits<float>::epsilon();
    W += grad_W;
    V += grad_V;
  }
}

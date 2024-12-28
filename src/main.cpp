#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

template <typename T> T seed_random() {
  std::random_device r;
  std::seed_seq seed{r(), r(), r(), r()};
  T ret{seed};
  return ret;
}

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

template <typename T, size_t N> struct alignas(64) vector {
  std::array<T, N> d;
  vector(std::nullopt_t) noexcept {}
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

  [[nodiscard]] constexpr size_t size() const noexcept { return N; }

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

enum class activation_function { sigmoid, square_sigmoid, square_plus };

template <typename T, activation_function Activation, activation_function FinalActivation> class neural_network {

  static constexpr std::array<size_t, 4> sizes{2, 4, 4, 2};
  template <size_t Layer> using matrix_t = matrix<T, sizes[Layer], sizes[Layer + 1]>;
  using matrix_tuple_t = std::tuple<matrix_t<0>, matrix_t<1>, matrix_t<2>>;
  matrix_tuple_t layers;
  matrix_tuple_t gradients;

  static constexpr size_t last_layer = std::tuple_size_v<matrix_tuple_t> - 1;

  template <size_t Layer> matrix_t<Layer> &get_layer() noexcept { return std::get<Layer>(layers); }
  template <size_t Layer> matrix_t<Layer> &get_gradient() noexcept { return std::get<Layer>(gradients); }

  template <size_t Layer> using vector_t = vector<T, sizes[Layer]>;
  using vector_tuple_t = std::tuple<vector_t<1>, vector_t<2>, vector_t<3>>;
  vector_tuple_t intermediate_results;
  vector_tuple_t intermediate_deltas;
  template <size_t Layer> vector_t<Layer + 1> &get_result() noexcept { return std::get<Layer>(intermediate_results); }
  template <size_t Layer> vector_t<Layer + 1> &get_delta() noexcept { return std::get<Layer>(intermediate_deltas); }

  T alpha = .5;

  std::vector<std::pair<vector_t<0>, vector_t<sizes.size() - 1>>> data;

  template <size_t layer, activation_function f> void apply_activation(T x, size_t i) noexcept {
    using enum activation_function;
    if constexpr (f == sigmoid) {
      const T e = 1 / (1 + std::exp(-x));
      get_result<layer>()[i] = e;
      get_delta<layer>()[i] = e * (1 - e);
    } else {
      const T x1 = x * x + 1;
      const T x2 = 1 / sqrt(x1);
      if constexpr (f == square_plus) {
        get_result<layer>()[i] = x1 * x2 + x;
        get_delta<layer>()[i] = x * x2 + 1;
      } else if constexpr (f == square_sigmoid) {
        get_result<layer>()[i] = x * x2;
        get_delta<layer>()[i] = x2 * x2 * x2;
      }
    }
  }

  template <size_t layer> void apply_activation(T x, size_t i) noexcept {
    if constexpr (layer == sizes.size() - 1)
      apply_activation<layer, FinalActivation>(x, i);
    else
      apply_activation<layer, Activation>(x, i);
  }

  template <size_t Layer = 0> void forward(size_t i) noexcept {
    const vector_t<Layer> &v_ = [this, i]() -> const vector_t<Layer> & {
      if constexpr (Layer)
        return get_result<Layer - 1>();
      else
        return data[i].first;
    }();

#if 1
    const T *v = v_.d.data();
    const T *m = get_layer<Layer>().d.data();
    T *o = get_result<Layer>().d.data();

    for (size_t i = sizes[Layer + 1]; i--;)
      *o++ = *v * *m++;
    ++v;

    for (size_t i = sizes[Layer] - 1; --i;) {
      o = get_result<Layer>().d.data();
      for (size_t j = sizes[Layer + 1]; j--;)
        *o++ += *v * *m++;
      ++v;
    }

    o = get_result<Layer>().d.data();
    for (size_t i = 0; i < sizes[Layer + 1]; i++)
      apply_activation<Layer>(*o++ + *v * *m++, i);
#else
    vector_t<Layer + 1> &o = get_result<Layer>();
    o = v_ * get_layer<Layer>();
    for (size_t i = 0; i < o.size(); i++)
      apply_activation<Layer>(o[i], i);
#endif

    if constexpr (Layer < last_layer)
      forward<Layer + 1>(-1);
  }

  template <size_t N> void init(std::mt19937_64 &r, std::normal_distribution<T> &d) {
    for (T &x : get_layer<N>().d)
      x = d(r);
    if constexpr (N)
      init<N - 1>(r, d);
  }

  template <size_t Layer = last_layer> void backward(size_t i, vector_t<Layer + 1> &grad_in) noexcept {
    grad_in *= get_delta<Layer>();
    if constexpr (Layer) {
      get_gradient<Layer>().add_outer_product(get_result<Layer - 1>(), grad_in);
      auto grad_out = get_layer<Layer>() * grad_in;
      backward<Layer - 1>(i, grad_out);
    } else {
      get_gradient<Layer>().add_outer_product(data[i].first, grad_in);
    }
  }

  template <size_t Layer = last_layer> void reset() {
    matrix_t<Layer> &m = get_gradient<Layer>();
    for (T &x : m.d)
      x = 0;
    if constexpr (Layer)
      reset<Layer - 1>();
  }

  template <size_t Layer = last_layer> void apply() {
    auto &out = get_layer<Layer>();
    const auto &in = get_gradient<Layer>();
    const T a = alpha / data.size();
    static_assert(in.d.size() == out.d.size());
    for (size_t i = 0; i < sizes[Layer] * sizes[Layer + 1]; i++)
      out.d[i] -= in.d[i] * a;
    if constexpr (Layer)
      apply<Layer - 1>();
  }

public:
  void init() {
    auto r = seed_random<std::mt19937_64>();
    std::normal_distribution<T> d{0, 1};
    init<sizes.size() - 2>(r, d);
  }

  static_assert(sizes.front() == 2);
  static_assert(sizes.back() == 2);

  void insert_pair(const vector_t<0> &x, const vector_t<sizes.size() - 1> &y) { data.emplace_back(x, y); }

  void train() {
    T e = 0;
    for (size_t i = 0; i < data.size(); i++) {
      forward<>(i);
      reset<>();
      vector_t<last_layer + 1> E = get_result<last_layer>() - data[i].second;
      e += E.err();
      E += E;
      backward<>(i, E);
      apply<>();
    }
    std::cout << e / data.size() << std::endl;
  }
};

using neural_network_t =
    neural_network<float, activation_function::square_sigmoid, activation_function::square_sigmoid>;

static float is_o(const vector<float, 2> &v) {
#if 1
  return v[0] * v[0] > v[1] ? 1 : -1;
#else
  const float mag = v[0] * v[0] + v[1] * v[1];
  return 1 < mag && mag < 2 ? 1 : -1;
#endif
}

static float is_x(const vector<float, 2> &v) {
  const float a = std::abs(v[0] + v[1]);
  const float b = std::abs(v[0] - v[1]);
  return a < 1 || b < 1 ? 1 : -1;
}

static void fill_xo(neural_network_t &nn) {
  auto r = seed_random<std::mt19937_64>();
  std::uniform_real_distribution<float> d{-4, 4};
  for (int i = 1000; i--;) {
    vector<float, 2> in;
    do {
      in[0] = d(r);
      in[1] = d(r);
    } while (in[0] * in[0] + in[1] * in[1] > 16);
    vector<float, 2> out{{is_o(in), is_x(in)}};
    nn.insert_pair(in, out);
  }
}

int main() {
  using matrix_t = matrix<float, 2, 2>;
  using vector_t = vector<float, 2>;

  // matrix_t W{{6, -3, -2, 5}};
  // matrix_t V{{1, -2, .25f, 2}};
  std::array<matrix_t, 2> layers{matrix_t{{1, 2, 3, 4}}, matrix_t{{-1, -2, -3, -4}}};
  const std::array<vector_t, 2> X{vector_t{{3, 1}}, vector_t{{-1, 4}}};
  const std::array<vector_t, 2> T{vector_t{{1, 0}}, vector_t{{0, 1}}};

  std::unique_ptr<neural_network_t> nn = std::make_unique<neural_network_t>();
  nn->init();
  fill_xo(*nn);
  for (;;)
    nn->train();

  float alpha = -1;

  for (int iteration = 0; std::isnormal(alpha); iteration++) {
    std::array<matrix_t, 2> grad;
    for (size_t i = 0; i < 2; i++) {
      vector_t H = X[i] * layers[0];
      H.actual_sigmoid();

      vector_t Y = H * layers[1];
      Y.actual_sigmoid();

      vector_t E = Y - T[i];
      if (!(iteration & (iteration - 1)))
        std::cout << (E * E).sum() << std::endl;

      vector_t grad_Y = Y * (vector_t{{1, 1}} - Y) * (E + E);
      grad[1].add_outer_product(H, grad_Y);

      vector_t grad_H = (H * (vector_t{{1, 1}} - H) * layers[1]) * grad_Y;
      grad[0].add_outer_product(X[i], grad_H);
    }

    grad[0] *= alpha;
    grad[1] *= alpha;
    alpha *= 1 - std::numeric_limits<float>::epsilon();
    layers[0] += grad[0];
    layers[1] += grad[1];
  }
}

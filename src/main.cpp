#include "matrix.hpp"
#include "vector.hpp"
#include <array>
#include <boost/mp11.hpp>
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

enum class activation_function { sigmoid, square_sigmoid, square_plus };

template <typename T, activation_function Activation, activation_function FinalActivation, size_t... Sizes>
class neural_network {
  using sizes = boost::mp11::mp_list_c<size_t, Sizes...>;

  template <size_t Layer>
  using matrix_t = matrix<T, boost::mp11::mp_at_c<sizes, Layer>::value, boost::mp11::mp_at_c<sizes, Layer + 1>::value>;
  template <typename Layer> using matrix_c = matrix_t<Layer::value>;
  using matrix_tuple_t = boost::mp11::mp_rename<
      boost::mp11::mp_transform<matrix_c, boost::mp11::mp_iota_c<boost::mp11::mp_size<sizes>::value - 1>>, std::tuple>;

  matrix_tuple_t layers;
  matrix_tuple_t gradients;

  static constexpr size_t last_layer = std::tuple_size_v<matrix_tuple_t> - 1;

  template <size_t Layer> matrix_t<Layer> &get_layer() noexcept { return std::get<Layer>(layers); }
  template <size_t Layer> matrix_t<Layer> &get_gradient() noexcept { return std::get<Layer>(gradients); }

  template <size_t Layer> using vector_t = vector<T, boost::mp11::mp_at_c<sizes, Layer>::value>;
  template <typename Size> using vector_c = vector<T, Size::value>;
  using vector_tuple_t =
      boost::mp11::mp_rename<boost::mp11::mp_transform<vector_c, boost::mp11::mp_pop_front<sizes>>, std::tuple>;

  vector_tuple_t intermediate_results;
  vector_tuple_t intermediate_deltas;
  template <size_t Layer> vector_t<Layer + 1> &get_result() noexcept { return std::get<Layer>(intermediate_results); }
  template <size_t Layer> vector_t<Layer + 1> &get_delta() noexcept { return std::get<Layer>(intermediate_deltas); }

  T alpha = .5;

  std::vector<std::pair<vector_t<0>, vector_t<last_layer + 1>>> data;

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
    if constexpr (layer == last_layer)
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

#if 0
    using boost::mp11::mp_at_c;
    const T *v = v_.d.data();
    const T *m = get_layer<Layer>().d.data();
    T *o = get_result<Layer>().d.data();

    for (size_t i = mp_at_c<sizes, Layer + 1>::value; i--;)
      *o++ = *v * *m++;
    ++v;

    for (size_t i = mp_at_c<sizes, Layer>::value - 1; --i;) {
      o = get_result<Layer>().d.data();
      for (size_t j = mp_at_c<sizes, Layer + 1>::value; j--;)
        *o++ += *v * *m++;
      ++v;
    }

    o = get_result<Layer>().d.data();
    for (size_t i = 0; i < mp_at_c<sizes, Layer + 1>::value; i++)
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

  template <size_t N = last_layer> void init(std::mt19937_64 &r, std::normal_distribution<T> &d) {
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
    for (size_t i = 0; i < out.d.size(); i++)
      out.d[i] -= in.d[i] * a;
    if constexpr (Layer)
      apply<Layer - 1>();
  }

public:
  void init() {
    auto r = seed_random<std::mt19937_64>();
    std::normal_distribution<T> d{0, 1};
    init<>(r, d);
  }

  void insert_pair(const vector_t<0> &x, const vector_t<last_layer + 1> &y) { data.emplace_back(x, y); }

  void train(bool do_error = true) {
    vector_t<last_layer + 1> e{};
    for (size_t i = 0; i < data.size(); i++) {
      forward<>(i);
      reset<>();
      vector_t<last_layer + 1> E = get_result<last_layer>() - data[i].second;
      if (do_error) {
        e += E * E;
      }
      E += E;
      backward<>(i, E);
      apply<>();
    }
    if (do_error) {
      e *= static_cast<T>(1) / static_cast<T>(data.size());
      std::cout << e.sum() << '\t' << e << std::endl;
    }
  }
};

using neural_network_t = neural_network<float, activation_function::square_sigmoid, activation_function::square_sigmoid,
                                        2, 16, 16, 16, 16, 3>;

static float is_o(const vector<float, 2> &v) {
#if 1
  return v[0] * v[0] > v[1] ? .5f : -.5f;
#else
  const float mag = v[0] * v[0] + v[1] * v[1];
  return 1 < mag && mag < 2 ? 1 : -1;
#endif
}

static float is_x(const vector<float, 2> &v) {
  const float a = std::abs(v[0] + v[1]);
  const float b = std::abs(v[0] - v[1]);
  return a < 1 || b < 1 ? .5f : -.5f;
}

static void fill_xo(neural_network_t &nn) {
  auto r = seed_random<std::mt19937_64>();
  std::uniform_real_distribution<float> d{-4, 4};
  for (int i = 1000000; i--;) {
    vector<float, 2> in;
    do {
      in[0] = d(r);
      in[1] = d(r);
    } while (in[0] * in[0] + in[1] * in[1] > 16);
    vector<float, 3> out{{is_o(in), is_x(in), ((in[0] > 0.0f) && (in[1] > 0.0f)) ? .5f : -.5f}};
    in *= .25f;
    nn.insert_pair(in, out);
  }
}

int main() {
  std::ios::sync_with_stdio(false);

#if 1
  std::unique_ptr<neural_network_t> nn = std::make_unique<neural_network_t>();
  nn->init();
  fill_xo(*nn);
  for (size_t i = 0;; i++)
    nn->train(!(i & (i - 1)));
#elif 1
  neural_network<float, activation_function::square_sigmoid, activation_function::square_sigmoid, 2, 2, 2> nn;
  nn.init();
  nn.insert_pair({{3, 1}}, {{1, -1}});
  nn.insert_pair({{-1, 4}}, {{-1, 1}});
  for (size_t i = 0;; i++)
    nn.train(!(i & (i - 1)));
#else
  using matrix_t = matrix<float, 2, 2>;
  using vector_t = vector<float, 2>;

  std::array<matrix_t, 2> layers{matrix_t{{1, 2, 3, 4}}, matrix_t{{-1, -2, -3, -4}}};
  const std::array<vector_t, 2> X{vector_t{{3, 1}}, vector_t{{-1, 4}}};
  const std::array<vector_t, 2> T{vector_t{{1, 0}}, vector_t{{0, 1}}};

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
#endif
}

#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include <boost/mp11.hpp>
#include <iostream>
#include <random>
#include <vector>

namespace nnla {

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
  static_assert(boost::mp11::mp_size<sizes>::value >= 2);

  template <size_t layer> [[nodiscard]] static constexpr size_t get_size() noexcept {
    return boost::mp11::mp_at_c<sizes, layer>::value;
  }

  template <size_t Layer>
  using matrix_t = matrix<T, boost::mp11::mp_at_c<sizes, Layer>::value, boost::mp11::mp_at_c<sizes, Layer + 1>::value>;
  template <typename Layer> using matrix_c = matrix_t<Layer::value>;
  using matrix_tuple_t = boost::mp11::mp_rename<
      boost::mp11::mp_transform<matrix_c, boost::mp11::mp_iota_c<boost::mp11::mp_size<sizes>::value - 1>>, std::tuple>;

  matrix_tuple_t layers;
  matrix_tuple_t gradients;

  static constexpr size_t last_layer = std::tuple_size_v<matrix_tuple_t> - 1;

  template <size_t Layer> matrix_t<Layer> &get_layer() noexcept { return std::get<Layer>(layers); }
  template <size_t Layer> const matrix_t<Layer> &get_layer() const noexcept { return std::get<Layer>(layers); }
  template <size_t Layer> matrix_t<Layer> &get_gradient() noexcept { return std::get<Layer>(gradients); }

  template <size_t Layer> using vector_t = vector<T, boost::mp11::mp_at_c<sizes, Layer>::value>;
  template <typename Size> using vector_c = vector<T, Size::value>;
  using vector_tuple_t =
      boost::mp11::mp_rename<boost::mp11::mp_transform<vector_c, boost::mp11::mp_pop_front<sizes>>, std::tuple>;

  vector_tuple_t intermediate_results;
  vector_tuple_t intermediate_deltas;
  vector_tuple_t biases_deltas;
  vector_tuple_t biases;
  template <size_t Layer> vector_t<Layer + 1> &get_result() noexcept { return std::get<Layer>(intermediate_results); }
  template <size_t Layer> vector_t<Layer + 1> &get_delta() noexcept { return std::get<Layer>(intermediate_deltas); }
  template <size_t Layer> vector_t<Layer + 1> &get_bias() noexcept { return std::get<Layer>(biases); }
  template <size_t Layer> vector_t<Layer + 1> &get_bias_delta() noexcept { return std::get<Layer>(biases); }
  template <size_t Layer> const vector_t<Layer + 1> &get_bias() const noexcept { return std::get<Layer>(biases); }

  static constexpr T fudge = 1;

  std::vector<vector_t<0>> data_in;
  std::vector<vector_t<last_layer + 1>> data_out;

  template <size_t layer> void apply_activation(vector_t<layer + 1> &res) const noexcept {
    using enum activation_function;
    static constexpr activation_function f = layer == last_layer ? FinalActivation : Activation;

    for (T &x : res) {
      if constexpr (f == sigmoid) {
        const T e = 1 / (1 + std::exp(-x));
        x = fudge * e;
      } else {
        const T x1 = x * x + 1;
        const T x3 = 1 / std::sqrt(x1);
        const T x2 = x1 * x3;
        if constexpr (f == square_plus) {
          x = x2 + fudge * x;
        } else if constexpr (f == square_sigmoid) {
          x = fudge * x * x3;
        }
      }
    }
  }

  template <size_t layer> void apply_activation_and_delta(const vector_t<layer + 1> &x) noexcept {
    vector_t<layer + 1> &res = get_result<layer>();
    vector_t<layer + 1> &delta = get_delta<layer>();

    [[assume(&res != &delta)]];

    using enum activation_function;
    static constexpr activation_function f = layer == last_layer ? FinalActivation : Activation;

    for (size_t i = 0; i < get_size<layer + 1>(); i++) {
      if constexpr (f == sigmoid) {
        const T e = 1 / (1 + std::exp(-x[i]));
        delta[i] = fudge * e * (1 - e);
        res[i] = fudge * e;
      } else {
        const T x1 = x[i] * x[i] + 1;
        const T x3 = 1 / std::sqrt(x1);
        const T x2 = x1 * x3;
        if constexpr (f == square_plus) {
          delta[i] = fudge + x[i] * x3;
          res[i] = x2 + fudge * x[i];
        } else if constexpr (f == square_sigmoid) {
          delta[i] = fudge * x3 * x3 * x3;
          res[i] = fudge * x[i] * x3;
        }
      }
    }
  }

  template <size_t Layer = 0> void forward(const vector_t<Layer> &in) noexcept {
    vector_t<Layer + 1> next = get_bias<Layer>();
    const auto &m = get_layer<Layer>();
    for (size_t i = 0; i < in.size; i++)
      for (size_t j = 0; j < next.size; j++)
        next[j] += in[i] * m[i][j];
    apply_activation_and_delta<Layer>(next);

    if constexpr (Layer < last_layer)
      forward<Layer + 1>(get_result<Layer>());
  }

  template <size_t Layer = 0> const vector_t<last_layer + 1> forward_use(const vector_t<Layer> &in) const noexcept {
    vector_t<Layer + 1> next = get_bias<Layer>();
    const auto &m = get_layer<Layer>();
    for (size_t i = 0; i < in.size; i++)
      for (size_t j = 0; j < next.size; j++)
        next[j] += in[i] * m[i][j];
    apply_activation<Layer>(next);

    if constexpr (Layer < last_layer)
      return forward_use<Layer + 1>(next);
    else
      return next;
  }

  template <size_t N = last_layer> void init(std::mt19937_64 &r, std::normal_distribution<T> &d) {
    for (T &x : get_layer<N>())
      x = d(r);
    if constexpr (N) {
      for (T &x : get_bias<N>())
        x = d(r);
      init<N - 1>(r, d);
    }
  }

  template <size_t Layer = last_layer>
  __attribute__((noinline)) void backward(size_t i, vector_t<Layer + 1> grad_in) noexcept {
    grad_in *= get_delta<Layer>();
    get_bias_delta<Layer>() += grad_in;
    if constexpr (Layer) {
      get_gradient<Layer>().add_outer_product(get_result<Layer - 1>(), grad_in);
      backward<Layer - 1>(i, get_layer<Layer>() * grad_in);
    } else {
      get_gradient<Layer>().add_outer_product(data_in[i], grad_in);
    }
  }

  template <size_t Layer = last_layer> void reset() {
    for (T &x : get_gradient<Layer>())
      x = 0;
    for (T &x : get_bias_delta<Layer>())
      x = 0;
    if constexpr (Layer)
      reset<Layer - 1>();
  }

  template <size_t Layer = last_layer> void apply() {
    auto out = get_layer<Layer>().begin();
    auto in = get_gradient<Layer>().begin();
    const T a = alpha / data_in.size();
    for (size_t i = get_layer<Layer>().size; i--;)
      *out++ -= *in++ * a;

    auto out_bias = get_bias<Layer>().begin();
    auto in_bias = get_bias_delta<Layer>().begin();
    for (size_t i = get_bias<Layer>().size; i--;)
      *out_bias++ -= *in_bias++ * a;

    if constexpr (Layer)
      apply<Layer - 1>();
  }

public:
  static constexpr T alpha = .5;

  void init() {
    auto r = seed_random<std::mt19937_64>();
    std::normal_distribution<T> d{0, 1};
    init<>(r, d);
  }

  void insert_pair(const vector_t<0> &x, const vector_t<last_layer + 1> &y) {
    data_in.emplace_back(x);
    data_out.emplace_back(y);
  }

  vector_t<last_layer + 1> train() noexcept {
    vector_t<last_layer + 1> e{};
    for (size_t i = 0; i < data_in.size(); i++) {
      [[likely]];
      forward<>(data_in[i]);
      reset<>();
      vector_t<last_layer + 1> E = get_result<last_layer>() - data_out[i];
      e += E * E;
      E += E;
      backward<>(i, E);
      apply<>();
    }
    e /= data_in.size();
    return e;
  }

  std::vector<vector_t<last_layer + 1>> forward(const std::vector<vector_t<0>> &in) const noexcept {
    std::vector<vector_t<last_layer + 1>> out;
    out.reserve(in.size());
    for (const auto &v : in)
      out.emplace_back(forward_use<>(v));
    return out;
  }
};

} // namespace nnla

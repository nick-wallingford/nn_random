#include "matrix.hpp"
#include "nn.hpp"
#include "vector.hpp"
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using neural_network_t =
    neural_network<float, activation_function::square_plus, activation_function::square_sigmoid, 2, 8, 16, 4>;

static constexpr float true_t = 1;
static constexpr float false_t = -1;
static constexpr float unsure_t = (true_t + false_t) / 2;

[[maybe_unused]] static float is_x(const vector<float, 2> &v) {
  const float a = std::abs(v[0] + v[1]);
  const float b = std::abs(v[0] - v[1]);
  return a < 1 || b < 1 ? true_t : false_t;
}

[[maybe_unused]] static float is_parabola(const vector<float, 2> &v) { return v[0] * v[0] < v[1] ? true_t : false_t; }

[[maybe_unused]] static float is_o(const vector<float, 2> &v) {
  const float mag = v[0] * v[0] + v[1] * v[1];
#if 1
  return std::sqrt(4.f) < mag && mag < std::sqrt(12.f);
#else
  return mag < std::sqrt(6.f) ? true_t : false_t;
#endif
}

[[maybe_unused]] static float is_and(const vector<float, 2> &v) { return v[0] > 0 && v[1] > 0 ? true_t : false_t; }

static void fill_xo(neural_network_t &nn) {
  auto r = seed_random<std::mt19937_64>();
  std::uniform_real_distribution<float> d{-4, 4};
  for (int i = 1000000; i--;) {
    vector<float, 2> in;
    do {
      in[0] = d(r);
      in[1] = d(r);
    } while (in[0] * in[0] + in[1] * in[1] > 16);
    vector<float, 4> out{{is_o(in), is_x(in), is_parabola(in), is_and(in)}};
    nn.insert_pair(in, out);
  }
}

static void print(const neural_network_t &nn) {
  std::vector<vector<float, 2>> in;
  for (float y = 4; y >= -4; y -= 16.f / 40.f)
    for (float x = -4; x <= 4; x += 8.f / 40.f)
      in.push_back(vector<float, 2>{{x, y}});
  const auto v = nn.forward(in);
  static constexpr std::array<std::string_view, 3> chars{"\033[31mX", "\033[32m0", "\033[0m"};
  for (size_t i = 0; i < v[0].d.size(); i++) {
    size_t j = 0;
    for (float y = 4; y >= -4; y -= 16.f / 40.f) {
      for (float x = -4; x <= 4; x += 8.f / 40.f)
        if (x * x + y * y > 16) {
          j++;
          std::cout << ' ';
        } else
          std::cout << chars[v[j++][i] > unsure_t];
      std::cout << chars[2] << '\n';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
}

int main() {
  std::ios::sync_with_stdio(false);

#if 1
  std::unique_ptr<neural_network_t> nn = std::make_unique<neural_network_t>();
  nn->init();
  fill_xo(*nn);
  for (size_t i = 0;; i++) {
    const bool doit = !(i & (i - 1));
    if (doit)
      print(*nn);
    nn->train(doit);
  }
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

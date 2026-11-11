// Inspired by hmmlearn's
// https://github.com/hmmlearn/hmmlearn/blob/e01a10e99df1042e4c1e6b7c822fd292dd37502f/ext/_hmmc.cpp
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using Size = py::ssize_t;
using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

static void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}

static Size product(std::initializer_list<Size> dimensions) {
  Size result = 1;
  for (Size dimension : dimensions) {
    require(dimension > 0, "model dimensions must be positive");
    if (result > std::numeric_limits<Size>::max() / dimension)
      throw std::overflow_error("parameter dimensions are too large");
    result *= dimension;
  }
  return result;
}

struct Dimensions {
  Size M, O, A, R;
  Size b, eo, er, b0, eo0, er0;
  Dimensions(Size memory, Size observations, Size actions, Size rewards)
      : M(memory), O(observations), A(actions), R(rewards),
        b(product({O, A, M, M})), eo(product({O, A, O, M})),
        er(product({O, A, R, M})), b0(product({A, M})), eo0(product({A, O})),
        er0(product({A, R})) {}
};

// NumPy owns each buffer; it is allocated before releasing the GIL and never
// resized during fitting. Numerical routines use cached C++ pointers/views.
struct Buffer {
  Doubles array;
  double *pointer;
  Size length;
  Buffer(std::initializer_list<Size> shape)
      : array(shape), pointer(array.mutable_data()), length(array.size()) {}
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) = default;
  Buffer &operator=(Buffer &&) = default;
  double *data() { return pointer; }
  const double *data() const { return pointer; }
  double *begin() { return pointer; }
  double *end() { return pointer + length; }
  const double *begin() const { return pointer; }
  const double *end() const { return pointer + length; }
  double &operator[](Size i) { return pointer[i]; }
  const double &operator[](Size i) const { return pointer[i]; }
};

// Internal layout keeps memory contiguous in every emission/transition lookup.
// Public layout remains B[m,o,a,m_next], EO[m,o,a,o_next], ER[m,o,a,r].
struct Heads {
  Buffer b, eo, er, b0, eo0, er0;
  explicit Heads(const Dimensions &d)
      : b({d.O, d.A, d.M, d.M}), eo({d.O, d.A, d.O, d.M}),
        er({d.O, d.A, d.R, d.M}), b0({d.A, d.M}), eo0({d.A, d.O}),
        er0({d.A, d.R}) {}
};

// Sufficient statistics use a memory-first layout. Parameters
// keep the lookup layout above; M-steps convert between the two explicitly.
struct Statistics {
  Buffer b, eo, er, b0, eo0, er0;
  explicit Statistics(const Dimensions &d)
      : b({d.M, d.M, d.O, d.A}), eo({d.M, d.O, d.A, d.O}),
        er({d.M, d.O, d.A, d.R}), b0({d.A, d.M}), eo0({d.A, d.O}),
        er0({d.A, d.R}) {}
};

static void parameter_counts(const Statistics &counts, Heads &result,
                             const Dimensions &d) {
  // Write contiguous lookup rows from the memory-first statistics.
  for (Size o = 0; o < d.O; ++o)
    for (Size a = 0; a < d.A; ++a) {
      const Size context = o * d.A + a;
      for (Size m = 0; m < d.M; ++m)
        for (Size next = 0; next < d.M; ++next)
          result.b[(context * d.M + m) * d.M + next] =
              counts.b[((m * d.M + next) * d.O + o) * d.A + a];
      for (Size on = 0; on < d.O; ++on)
        for (Size m = 0; m < d.M; ++m)
          result.eo[(context * d.O + on) * d.M + m] =
              counts.eo[((m * d.O + o) * d.A + a) * d.O + on];
      for (Size r = 0; r < d.R; ++r)
        for (Size m = 0; m < d.M; ++m)
          result.er[(context * d.R + r) * d.M + m] =
              counts.er[((m * d.O + o) * d.A + a) * d.R + r];
    }
  std::copy(counts.b0.begin(), counts.b0.end(), result.b0.begin());
  std::copy(counts.eo0.begin(), counts.eo0.end(), result.eo0.begin());
  std::copy(counts.er0.begin(), counts.er0.end(), result.er0.begin());
}

struct Data {
  Indices actions, observations, rewards, lengths;
  Size N, H;
  Data(Indices a, Indices o, Indices r, Indices l, const Dimensions &d)
      : actions(a), observations(o), rewards(r), lengths(l) {
    require(a.ndim() == 2 && o.ndim() == 2 && r.ndim() == 2 && l.ndim() == 1,
            "expected three (N,H) arrays and a length vector");
    N = a.shape(0);
    H = a.shape(1);
    require(H > 0 && o.shape(0) == N && r.shape(0) == N && o.shape(1) == H &&
                r.shape(1) == H && l.size() == N,
            "trajectory arrays/lengths have incompatible shapes");
    for (Size i = 0; i < N; ++i) {
      require(l.data()[i] >= 0 && l.data()[i] <= H,
              "trajectory length must be between zero and H");
      // Padding is never read by the fitter.
      for (Size t = 0; t < l.data()[i]; ++t) {
        Size cell = i * H + t;
        require(a.data()[cell] >= 0 && a.data()[cell] < d.A,
                "action index out of range");
        require(o.data()[cell] >= 0 && o.data()[cell] < d.O,
                "observation index out of range");
        require(r.data()[cell] >= 0 && r.data()[cell] < d.R,
                "reward index out of range");
      }
    }
  }
};

static void normalize_row(double *row, Size count, Size stride,
                          double pseudocount) {
  double total = 0;
  for (Size i = 0; i < count; ++i)
    total += (row[i * stride] += pseudocount);
  if (!std::isfinite(total)) {
    double largest = 0;
    for (Size i = 0; i < count; ++i)
      largest = std::max(largest, row[i * stride]);
    total = 0;
    for (Size i = 0; i < count; ++i)
      total += (row[i * stride] /= largest);
  }
  if (total == 0) {
    for (Size i = 0; i < count; ++i)
      row[i * stride] = 1.0 / count;
    return;
  }
  const double reciprocal = 1.0 / total;
  if (std::isfinite(reciprocal)) {
    for (Size i = 0; i < count; ++i)
      row[i * stride] *= reciprocal;
  } else {
    for (Size i = 0; i < count; ++i)
      row[i * stride] /= total;
  }
}

static void normalize_heads(Heads &result, const Dimensions &d,
                            double pseudocount) {
  for (Size o = 0; o < d.O; ++o)
    for (Size a = 0; a < d.A; ++a)
      for (Size m = 0; m < d.M; ++m) {
        normalize_row(result.b.data() + ((o * d.A + a) * d.M + m) * d.M, d.M, 1,
                      pseudocount);
        normalize_row(result.eo.data() + (o * d.A + a) * d.O * d.M + m, d.O,
                      d.M, pseudocount);
        normalize_row(result.er.data() + (o * d.A + a) * d.R * d.M + m, d.R,
                      d.M, pseudocount);
      }
  for (Size a = 0; a < d.A; ++a) {
    normalize_row(result.b0.data() + a * d.M, d.M, 1, pseudocount);
    normalize_row(result.eo0.data() + a * d.O, d.O, 1, pseudocount);
    normalize_row(result.er0.data() + a * d.R, d.R, 1, pseudocount);
  }
}

static void maximize(const Statistics &counts, Heads &result,
                     const Dimensions &d, double pseudocount) {
  parameter_counts(counts, result, d);
  normalize_heads(result, d, pseudocount);
}

static double log_prior(const Heads &theta, double pseudocount) {
  if (pseudocount == 0)
    return 0;
  double result = 0;
  for (auto *head :
       {&theta.b, &theta.eo, &theta.er, &theta.b0, &theta.eo0, &theta.er0})
    for (double probability : *head)
      result += std::log(probability);
  return pseudocount * result;
}

static Size random_label(std::mt19937_64 &rng, uint64_t bound,
                         uint64_t threshold) {
  // Fixed unbiased mapping, independent of standard-library distribution code.
  if (bound == 1)
    return 0;
  uint64_t bits;
  do {
    bits = rng();
  } while (bits < threshold);
  return static_cast<Size>((bound & (bound - 1)) == 0 ? bits & (bound - 1)
                                                      : bits % bound);
}

static void initialize(const Data &data, const Dimensions &d, uint64_t seed,
                       double pseudocount, Statistics &counts, Heads &theta) {
  for (auto *head : {&counts.b, &counts.eo, &counts.er, &counts.b0, &counts.eo0,
                     &counts.er0})
    std::fill(head->begin(), head->end(), 0.0);
  std::mt19937_64 rng(seed);
  const uint64_t bound = static_cast<uint64_t>(d.M),
                 threshold = (uint64_t(0) - bound) % bound;
  std::vector<Size> labels(data.H);
  for (Size i = 0; i < data.N; ++i) {
    Size length = data.lengths.data()[i], start = i * data.H;
    if (length == 0)
      continue;
    for (Size t = 0; t < length; ++t)
      labels[t] = random_label(rng, bound, threshold);
    labels[0] = 0;
    for (Size t = 0; t < length; ++t) {
      Size a = data.actions.data()[start + t],
           on = data.observations.data()[start + t],
           r = data.rewards.data()[start + t];
      if (t == 0) {
        counts.eo0[a * d.O + on] += 1;
        counts.er0[a * d.R + r] += 1;
        if (t + 1 < length)
          counts.b0[a * d.M + labels[t + 1]] += 1;
      } else {
        Size m = labels[t], o = data.observations.data()[start + t - 1];
        counts.eo[((m * d.O + o) * d.A + a) * d.O + on] += 1;
        counts.er[((m * d.O + o) * d.A + a) * d.R + r] += 1;
        if (t + 1 < length)
          counts.b[((m * d.M + labels[t + 1]) * d.O + o) * d.A + a] += 1;
      }
    }
  }
  maximize(counts, theta, d, pseudocount);
}

static Heads from_python(const py::tuple &arrays, const Dimensions &d,
                         double pseudocount) {
  require(arrays.size() == 6,
          "warm start must supply all six parameter arrays");
  const std::vector<std::vector<Size>> shapes = {
      {d.M, d.O, d.A, d.M}, {d.M, d.O, d.A, d.O}, {d.M, d.O, d.A, d.R},
      {d.A, d.M},           {d.A, d.O},           {d.A, d.R}};
  std::vector<Doubles> values;
  for (Size k = 0; k < 6; ++k) {
    Doubles value = Doubles::ensure(arrays[k]);
    require(value && value.ndim() == static_cast<Size>(shapes[k].size()),
            "invalid warm-start parameter dimensions");
    for (Size axis = 0; axis < value.ndim(); ++axis)
      require(value.shape(axis) == shapes[k][axis],
              "warm-start parameter shape does not match the problem/memory");
    Size columns = shapes[k].back();
    for (Size start = 0; start < value.size(); start += columns) {
      double total = 0;
      for (Size column = 0; column < columns; ++column) {
        double p = value.data()[start + column];
        require(std::isfinite(p) && p >= 0,
                "warm-start probabilities must be finite and nonnegative");
        total += p;
      }
      require(std::abs(total - 1) <= 1e-10,
              "warm-start probability rows must sum to one");
    }
    values.push_back(std::move(value));
  }
  Heads theta(d);
  for (Size m = 0; m < d.M; ++m)
    for (Size o = 0; o < d.O; ++o)
      for (Size a = 0; a < d.A; ++a) {
        for (Size n = 0; n < d.M; ++n)
          theta.b[((o * d.A + a) * d.M + m) * d.M + n] =
              values[0].data()[((m * d.O + o) * d.A + a) * d.M + n];
        for (Size on = 0; on < d.O; ++on)
          theta.eo[((o * d.A + a) * d.O + on) * d.M + m] =
              values[1].data()[((m * d.O + o) * d.A + a) * d.O + on];
        for (Size r = 0; r < d.R; ++r)
          theta.er[((o * d.A + a) * d.R + r) * d.M + m] =
              values[2].data()[((m * d.O + o) * d.A + a) * d.R + r];
      }
  std::copy_n(values[3].data(), d.b0, theta.b0.data());
  std::copy_n(values[4].data(), d.eo0, theta.eo0.data());
  std::copy_n(values[5].data(), d.er0, theta.er0.data());
  if (pseudocount > 0) {
    for (auto *head :
         {&theta.b, &theta.eo, &theta.er, &theta.b0, &theta.eo0, &theta.er0})
      for (double &probability : *head)
        probability = std::max(probability, 1e-30);
    normalize_heads(theta, d, 0);
  }
  return theta;
}

static py::tuple to_python(const Heads &theta, const Dimensions &d) {
  py::array_t<double> b({d.M, d.O, d.A, d.M}), eo({d.M, d.O, d.A, d.O}),
      er({d.M, d.O, d.A, d.R});
  py::array_t<double> b0({d.A, d.M}), eo0({d.A, d.O}), er0({d.A, d.R});
  for (Size m = 0; m < d.M; ++m)
    for (Size o = 0; o < d.O; ++o)
      for (Size a = 0; a < d.A; ++a) {
        for (Size n = 0; n < d.M; ++n)
          b.mutable_data()[((m * d.O + o) * d.A + a) * d.M + n] =
              theta.b[((o * d.A + a) * d.M + m) * d.M + n];
        for (Size on = 0; on < d.O; ++on)
          eo.mutable_data()[((m * d.O + o) * d.A + a) * d.O + on] =
              theta.eo[((o * d.A + a) * d.O + on) * d.M + m];
        for (Size r = 0; r < d.R; ++r)
          er.mutable_data()[((m * d.O + o) * d.A + a) * d.R + r] =
              theta.er[((o * d.A + a) * d.R + r) * d.M + m];
      }
  std::copy_n(theta.b0.data(), d.b0, b0.mutable_data());
  std::copy_n(theta.eo0.data(), d.eo0, eo0.mutable_data());
  std::copy_n(theta.er0.data(), d.er0, er0.mutable_data());
  return py::make_tuple(b, eo, er, b0, eo0, er0);
}

// Scaled forward-backward E-step with fixed-size memory loops.
template <Size fixed_memory>
static std::pair<double, Statistics>
expectation(const Heads &theta, const Data &data, const Dimensions &d) {
  Statistics counts(d);
  auto B = theta.b.array.unchecked<4>();
  auto Eo = theta.eo.array.unchecked<4>();
  auto Er = theta.er.array.unchecked<4>();
  auto B_init = theta.b0.array.unchecked<2>();
  auto Eo_init = theta.eo0.array.unchecked<2>();
  auto Er_init = theta.er0.array.unchecked<2>();
  auto B_num = counts.b.array.mutable_unchecked<4>();
  auto Eo_num = counts.eo.array.mutable_unchecked<4>();
  auto Er_num = counts.er.array.mutable_unchecked<4>();
  auto B_init_num = counts.b0.array.mutable_unchecked<2>();
  auto Eo_init_num = counts.eo0.array.mutable_unchecked<2>();
  auto Er_init_num = counts.er0.array.mutable_unchecked<2>();
  auto obs = data.observations.unchecked<2>();
  auto act = data.actions.unchecked<2>();
  auto reward = data.rewards.unchecked<2>();
  auto lengths = data.lengths.unchecked<1>();
  const Size M = fixed_memory == 0 ? d.M : fixed_memory;
  const Size H = data.H;
  auto alpha = std::unique_ptr<double[]>(new double[product({H, M})]);
  auto emission = std::unique_ptr<double[]>(new double[product({H, M})]);
  auto scale = std::unique_ptr<double[]>(new double[H]);
  auto beta = std::unique_ptr<double[]>(new double[M]);
  auto next_beta = std::unique_ptr<double[]>(new double[M]);
  auto weighted = std::unique_ptr<double[]>(new double[M]);
  double likelihood;
  {
    py::gil_scoped_release release;
    likelihood = [&]() -> double {
      const Size M = fixed_memory == 0 ? d.M : fixed_memory;
      for (auto *head : {&counts.b, &counts.eo, &counts.er, &counts.b0,
                         &counts.eo0, &counts.er0})
        std::fill(head->begin(), head->end(), 0.0);
      double log_likelihood = 0;
      for (py::ssize_t i = 0; i < act.shape(0); ++i) {
        const auto L = lengths(i);
        if (L == 0)
          continue;
        const auto a0 = act(i, 0);
        Eo_init_num(a0, obs(i, 0)) += 1.0;
        Er_init_num(a0, reward(i, 0)) += 1.0;
        log_likelihood += std::log(Eo_init(a0, obs(i, 0))) +
                          std::log(Er_init(a0, reward(i, 0)));
        if (L == 1)
          continue;
        for (py::ssize_t t = 1; t < L; ++t) {
          const auto o = obs(i, t - 1), a = act(i, t);
          for (py::ssize_t m = 0; m < M; ++m) {
            emission[t * M + m] =
                Eo(o, a, obs(i, t), m) * Er(o, a, reward(i, t), m);
            alpha[t * M + m] = 0.0;
          }
          if (t == 1) {
            for (py::ssize_t m = 0; m < M; ++m)
              alpha[t * M + m] = B_init(a0, m);
          } else {
            for (py::ssize_t m = 0; m < M; ++m)
              for (py::ssize_t mp = 0; mp < M; ++mp)
                alpha[t * M + mp] += alpha[(t - 1) * M + m] *
                                     B(obs(i, t - 2), act(i, t - 1), m, mp);
          }
          scale[t] = 0.0;
          for (py::ssize_t m = 0; m < M; ++m) {
            alpha[t * M + m] *= emission[t * M + m];
            scale[t] += alpha[t * M + m];
          }
          if (scale[t] == 0.0)
            return -std::numeric_limits<double>::infinity();
          for (py::ssize_t m = 0; m < M; ++m)
            alpha[t * M + m] /= scale[t];
          log_likelihood += std::log(scale[t]);
        }
        // The unused transition after the final emission integrates to one.
        // Accumulate routing counts only when a successor emission exists.
        std::fill_n(beta.get(), M, 1.0);
        for (py::ssize_t t = L - 1; t > 0; --t) {
          const auto o = obs(i, t - 1), a = act(i, t);
          double total = 0.0;
          for (py::ssize_t m = 0; m < M; ++m)
            total += alpha[t * M + m] * beta[m];
          if (total == 0.0)
            return -std::numeric_limits<double>::infinity();
          for (py::ssize_t m = 0; m < M; ++m) {
            const double posterior = alpha[t * M + m] * beta[m] / total;
            Eo_num(m, o, a, obs(i, t)) += posterior;
            Er_num(m, o, a, reward(i, t)) += posterior;
            weighted[m] = emission[t * M + m] * beta[m] / scale[t];
          }
          if (t == 1) {
            for (py::ssize_t m = 0; m < M; ++m)
              B_init_num(a0, m) += B_init(a0, m) * weighted[m];
          } else {
            for (py::ssize_t m = 0; m < M; ++m) {
              next_beta[m] = 0.0;
              for (py::ssize_t mp = 0; mp < M; ++mp) {
                const double contribution =
                    B(obs(i, t - 2), act(i, t - 1), m, mp) * weighted[mp];
                next_beta[m] += contribution;
                B_num(m, mp, obs(i, t - 2), act(i, t - 1)) +=
                    alpha[(t - 1) * M + m] * contribution;
              }
            }
            std::copy_n(next_beta.get(), M, beta.get());
          }
        }
      }
      if (!std::isfinite(log_likelihood))
        return -std::numeric_limits<double>::infinity();
      for (const auto *head : {&counts.b, &counts.eo, &counts.er, &counts.b0,
                               &counts.eo0, &counts.er0})
        for (double value : *head)
          if (!std::isfinite(value))
            return -std::numeric_limits<double>::infinity();
      return log_likelihood;
    }();
  }
  // NumPy owners are moved only after the GIL has been restored.
  return {likelihood, std::move(counts)};
}

static std::pair<double, Statistics>
expectation(const Heads &theta, const Data &data, const Dimensions &d) {
  if (d.M == 4)
    return expectation<4>(theta, data, d);
  if (d.M == 8)
    return expectation<8>(theta, data, d);
  return expectation<0>(theta, data, d);
}

static py::tuple fit(Indices actions, Indices observations, Indices rewards,
                     Indices lengths, Size memory, Size num_observations,
                     Size num_actions, Size num_rewards, Size max_iterations,
                     double tolerance, double pseudocount, uint64_t seed,
                     py::object initial) {
  require(max_iterations >= 0, "max_iterations must be nonnegative");
  require(std::isfinite(tolerance) && tolerance >= 0,
          "tolerance must be finite and nonnegative");
  require(std::isfinite(pseudocount) && pseudocount >= 0,
          "pseudocount must be finite and nonnegative");
  Dimensions d(memory, num_observations, num_actions, num_rewards);
  Data data(actions, observations, rewards, lengths, d);
  Heads theta(d);
  const bool cold = initial.is_none();
  if (!cold)
    theta = from_python(initial.cast<py::tuple>(), d, pseudocount);
  if (cold) {
    Statistics labeled_counts(d);
    // The arrays are allocated above; this scope performs only native math.
    py::gil_scoped_release release;
    initialize(data, d, seed, pseudocount, labeled_counts, theta);
  }
  auto [likelihood, counts] = expectation(theta, data, d);
  double objective;
  {
    py::gil_scoped_release release;
    objective = likelihood + log_prior(theta, pseudocount);
  }
  std::vector<double> likelihoods{likelihood}, objectives{objective};
  Size iterations = 0;
  std::string reason =
      std::isfinite(objective) ? "iteration_limit" : "numerical_failure";
  while (iterations < max_iterations && reason == "iteration_limit") {
    Heads candidate(d);
    {
      py::gil_scoped_release release;
      maximize(counts, candidate, d, pseudocount);
    }
    auto [next_likelihood, next_counts] = expectation(candidate, data, d);
    double next_objective;
    {
      py::gil_scoped_release release;
      next_objective = next_likelihood + log_prior(candidate, pseudocount);
    }
    if (!std::isfinite(next_objective)) {
      reason = "numerical_failure";
      break;
    }
    bool converged = std::abs(next_objective - objective) <=
                     tolerance * std::max(std::abs(objective), 1.0);
    theta = std::move(candidate);
    counts = std::move(next_counts);
    likelihood = next_likelihood;
    objective = next_objective;
    likelihoods.push_back(likelihood);
    objectives.push_back(objective);
    ++iterations;
    if (converged)
      reason = "converged";
  }
  return py::make_tuple(to_python(theta, d), iterations, likelihoods,
                        objectives, reason);
}

PYBIND11_MODULE(_em, module) {
#if defined(__clang__)
  module.attr("compiler") = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
  module.attr("compiler") = std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
  module.attr("compiler") = std::string("MSVC ") + std::to_string(_MSC_VER);
#else
  module.attr("compiler") = "unknown";
#endif
  module.doc() = "Finite-horizon IOHMM EM fit";
  module.def("fit", &fit);
}

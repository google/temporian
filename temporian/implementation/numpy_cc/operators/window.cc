#include <assert.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace py = pybind11;

typedef py::array_t<double> ArrayD;
typedef py::array_t<float> ArrayF;

template <typename T>
using ArrayRef = py::detail::unchecked_reference<T, 1>;

typedef size_t Idx;

// NOTE: accumulate() is overloaded for the 4 possible combinations of:
// - with or without external sampling
// - with constant or variable window length

// TODO: refactor to avoid code duplication where possible.

// No external sampling, constant window length
template <typename INPUT, typename OUTPUT, typename TAccumulator>
py::array_t<OUTPUT> accumulate(const ArrayD &evset_timestamps,
                               const py::array_t<INPUT> &evset_values,
                               const double window_length) {
  // Input size
  const size_t n_event = evset_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<OUTPUT>(n_event);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = evset_timestamps.unchecked<1>();
  auto v_values = evset_values.template unchecked<1>();

  TAccumulator accumulator(v_values);

  // Index of the first value in the window.
  size_t begin_idx = 0;
  // Index of the first value outside the window.
  size_t end_idx = 0;

  while (end_idx < n_event) {
    // Note: We accumulate values in (t-window_length, t] with t=
    // v_timestamps[end_idx], and there may be several contiguous equal
    // values in v_timestamps.

    // Add all values with same timestamp as the current one.
    accumulator.Add(end_idx);
    const auto current_ts = v_timestamps[end_idx];
    size_t first_diff_ts_idx = end_idx + 1;
    while (first_diff_ts_idx < n_event &&
           v_timestamps[first_diff_ts_idx] == current_ts) {
      accumulator.Add(first_diff_ts_idx);
      first_diff_ts_idx++;
    }

    // Remove all values that no longer belong to the window.
    while (begin_idx < n_event &&
           // Compare both sides around ~0 to get maximum float resolution
           v_timestamps[end_idx] - v_timestamps[begin_idx] >= window_length) {
      accumulator.Remove(begin_idx);
      begin_idx++;
    }

    // Set current value of window to all values with the same timestamp.
    const auto result = accumulator.Result();
    for (size_t i = end_idx; i < first_diff_ts_idx; i++) {
      v_output[i] = result;
    }

    // Move pointer to the index of the last value with the same timestamp.
    end_idx = first_diff_ts_idx;
  }

  return output;
}

// External sampling, constant window length
template <typename INPUT, typename OUTPUT, typename TAccumulator>
py::array_t<OUTPUT> accumulate(const ArrayD &evset_timestamps,
                               const py::array_t<INPUT> &evset_values,
                               const ArrayD &sampling_timestamps,
                               const double window_length) {
  // Input size
  const size_t n_event = evset_timestamps.shape(0);
  const size_t n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<OUTPUT>(n_sampling);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = evset_timestamps.unchecked<1>();
  auto v_values = evset_values.template unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();

  TAccumulator accumulator(v_values);

  size_t begin_idx = 0;
  size_t end_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto right_limit = v_sampling[sampling_idx];

    while (end_idx < n_event && v_timestamps[end_idx] <= right_limit) {
      accumulator.Add(end_idx);
      end_idx++;
    }

    while (begin_idx < n_event &&
           // Compare both sides around ~0 to get maximum float resolution
           v_sampling[sampling_idx] - v_timestamps[begin_idx] >=
               window_length) {
      accumulator.Remove(begin_idx);
      begin_idx++;
    }

    v_output[sampling_idx] = accumulator.Result();
  }

  return output;
}

bool begin_moved_forward(const double ts, const double prev_ts,
                         const double window_length,
                         const double prev_window_length) {
  return ts - prev_ts - (window_length - prev_window_length) > 0;
}

// No external sampling, variable window length
template <typename INPUT, typename OUTPUT, typename TAccumulator>
py::array_t<OUTPUT> accumulate(const ArrayD &evset_timestamps,
                               const py::array_t<INPUT> &evset_values,
                               const ArrayD &window_length) {
  // Input size
  const size_t n_event = evset_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<OUTPUT>(n_event);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = evset_timestamps.unchecked<1>();
  auto v_values = evset_values.template unchecked<1>();
  auto v_window_length = window_length.unchecked<1>();

  assert(v_timestamps.shape(0) == v_window_length.shape(0));
  assert(v_timestamps.shape(0) == v_values.shape(0));

  TAccumulator accumulator(v_values);

  // Index of the first value in the window.
  size_t begin_idx = 0;
  // Index of the first value outside the window.
  size_t end_idx = 0;

  // Note that end_idx might get ahead of idx if there are several values with
  // same timestamp in v_timestamps. We can't group these all together like we
  // do in the constant window case because they might have different window
  // lengths and therefore different output values.
  for (size_t idx = 0; idx < n_event; idx++) {
    // Note: We accumulate values in (t-window_length, t] with t=
    // v_timestamps[end_idx], and there may be several contiguous equal
    // values in v_timestamps.
    const auto curr_ts = v_timestamps[idx];
    auto curr_window_length = v_window_length[idx];

    if (std::isnan(curr_window_length)) {
      curr_window_length = 0;
    }

    while (end_idx < n_event && v_timestamps[end_idx] <= curr_ts) {
      accumulator.Add(end_idx);
      end_idx++;
    }

    // Move window's left limit forwards or backwards.
    if (idx == 0 ||
        begin_moved_forward(curr_ts, v_timestamps[idx - 1], curr_window_length,
                            v_window_length[idx - 1])) {
      // Window's beginning moved forwards.
      while (begin_idx < n_event &&
             v_timestamps[idx] - v_timestamps[begin_idx] >=
                 curr_window_length) {
        accumulator.Remove(begin_idx);
        begin_idx++;
      }
    } else {
      // Window's beginning moved backwards.
      // Note < instead of <= to respect (] window boundaries.
      while (begin_idx > 0 && v_timestamps[idx] - v_timestamps[begin_idx - 1] <
                                  curr_window_length) {
        begin_idx--;
        accumulator.AddLeft(begin_idx);
      }
    }

    v_output[idx] = accumulator.Result();
  }

  return output;
}

// External sampling, variable window length
template <typename INPUT, typename OUTPUT, typename TAccumulator>
py::array_t<OUTPUT> accumulate(const ArrayD &evset_timestamps,
                               const py::array_t<INPUT> &evset_values,
                               const ArrayD &sampling_timestamps,
                               const ArrayD &window_length) {
  // Input size
  const size_t n_event = evset_timestamps.shape(0);
  const size_t n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<OUTPUT>(n_sampling);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = evset_timestamps.unchecked<1>();
  auto v_values = evset_values.template unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();
  auto v_window_length = window_length.unchecked<1>();

  assert(v_timestamps.shape(0) == v_values.shape(0));
  assert(v_sampling.shape(0) == v_window_length.shape(0));

  TAccumulator accumulator(v_values);

  size_t begin_idx = 0;
  size_t end_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto right_limit = v_sampling[sampling_idx];
    auto curr_window_length = v_window_length[sampling_idx];

    if (std::isnan(curr_window_length)) {
      curr_window_length = 0;
    }

    while (end_idx < n_event && v_timestamps[end_idx] <= right_limit) {
      accumulator.Add(end_idx);
      end_idx++;
    }

    // Move window's left limit forwards or backwards.
    if (sampling_idx == 0 ||
        begin_moved_forward(right_limit, v_sampling[sampling_idx - 1],
                            curr_window_length,
                            v_window_length[sampling_idx - 1])) {
      // Window's beginning moved forwards.
      while (begin_idx < n_event &&
             right_limit - v_timestamps[begin_idx] >= curr_window_length) {
        accumulator.Remove(begin_idx);
        begin_idx++;
      }
    } else {
      // Window's beginning moved backwards.
      // Note < instead of <= to respect (] window boundaries.
      while (begin_idx > 0 &&
             right_limit - v_timestamps[begin_idx - 1] < curr_window_length) {
        begin_idx--;
        accumulator.AddLeft(begin_idx);
      }
    }

    v_output[sampling_idx] = accumulator.Result();
  }

  return output;
}

// Note: We only use inheritance to compile check the code.
template <typename INPUT, typename OUTPUT>
struct Accumulator {
  Accumulator(const ArrayRef<INPUT> &values) : values(values) {}

  virtual ~Accumulator() = default;
  virtual void Add(Idx idx) = 0;
  virtual void Remove(Idx idx) = 0;
  virtual OUTPUT Result() = 0;

  // // Add a value to left of the window. Relevant in deque-based accumulators.
  virtual void AddLeft(Idx idx) { return Add(idx); }

  ArrayRef<INPUT> values;
};

template <typename INPUT, typename OUTPUT>
struct SimpleMovingAverageAccumulator : public Accumulator<INPUT, OUTPUT> {
  SimpleMovingAverageAccumulator(const ArrayRef<INPUT> &values)
      : Accumulator<INPUT, OUTPUT>(values) {}

  void Add(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values += value;
    num_values++;
  }

  void Remove(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values -= value;
    num_values--;
  }

  OUTPUT Result() override {
    return (num_values > 0) ? (sum_values / num_values)
                            : std::numeric_limits<OUTPUT>::quiet_NaN();
  }

  // TODO(gbm): Increase precision of accumulator.

  // Sum of the values in the rolling window (RW).
  double sum_values = 0;
  // Number of values in the RW.
  int num_values = 0;
};

template <typename INPUT, typename OUTPUT>
struct MovingStandardDeviationAccumulator : Accumulator<INPUT, OUTPUT> {
  MovingStandardDeviationAccumulator(const ArrayRef<INPUT> &values)
      : Accumulator<INPUT, OUTPUT>(values) {}

  void Add(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values += value;
    sum_square_values += value * value;
    num_values++;
  }

  void Remove(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values -= value;
    sum_square_values -= value * value;
    num_values--;
  }

  OUTPUT Result() override {
    if (num_values == 0) {
      return std::numeric_limits<OUTPUT>::quiet_NaN();
    }
    const auto mean = sum_values / num_values;
    return sqrt(sum_square_values / num_values - mean * mean);
  }

  // Sum of the values in the rolling window (RW).
  double sum_values = 0;
  double sum_square_values = 0;
  // Number of values in the RW.
  int num_values = 0;
};

template <typename OUTPUT>
struct MovingCountAccumulator : Accumulator<double, OUTPUT> {
  MovingCountAccumulator(const ArrayRef<double> &values)
      : Accumulator<double, OUTPUT>(values) {}

  void Add(Idx idx) override {
    static_assert(std::is_same<OUTPUT, int32_t>::value,
                  "OUTPUT must be int32_t");
    num_values++;
  }

  void Remove(Idx idx) override { num_values--; }

  OUTPUT Result() override { return num_values; }

  // Number of values in the RW.
  int32_t num_values = 0;
};

template <typename INPUT, typename OUTPUT>
struct MovingSumAccumulator : Accumulator<INPUT, OUTPUT> {
  MovingSumAccumulator(const ArrayRef<INPUT> &values)
      : Accumulator<INPUT, OUTPUT>(values) {}

  void Add(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values += value;
  }

  void Remove(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (std::isnan(value)) {
        return;
      }
    }
    sum_values -= value;
  }

  OUTPUT Result() override { return sum_values; }

  // Sum of the values in the rolling window (RW).
  double sum_values = 0;
};

template <typename INPUT, typename OUTPUT>
struct MovingExtremumAccumulator : Accumulator<INPUT, OUTPUT> {
  MovingExtremumAccumulator(const ArrayRef<INPUT> &values)
      : Accumulator<INPUT, OUTPUT>(values) {}

  virtual ~MovingExtremumAccumulator() = default;

  virtual bool Compare(INPUT a, INPUT b) = 0;

  void Add(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (!(value == value)) {
        return;
      }
    }

    while (!best_indices.empty() &&
           !Compare(Accumulator<INPUT, OUTPUT>::values[best_indices.back()],
                    value)) {
      best_indices.pop_back();
    }
    best_indices.push_back(idx);
  }

  void AddLeft(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];

    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (!(value == value)) {
        return;
      }
    }

    if (best_indices.empty()) {
      best_indices.push_back(idx);
    } else {
      if (Compare(value,
                  Accumulator<INPUT, OUTPUT>::values[best_indices.front()])) {
        best_indices.push_front(idx);
      }
    }
  }

  void Remove(Idx idx) override {
    const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
    if constexpr (std::numeric_limits<INPUT>::has_quiet_NaN) {
      if (!(value == value)) {
        return;
      }
    }

    assert(!best_indices.empty());
    assert(best_indices.front() >= idx);
    if (best_indices.front() == idx) {
      best_indices.pop_front();
    }
  }

  OUTPUT Result() override {
    if (best_indices.empty()) {
      return std::numeric_limits<OUTPUT>::quiet_NaN();
    } else {
      return Accumulator<INPUT, OUTPUT>::values[best_indices.front()];
    }
  }

  // Subset of indexes of the observations in the window, sorted by index value,
  // and such that index "j" is not part of "best_indices" if there is another
  // index "i" in "best_indices" such that "i>j" (i is after j) and the value of
  // i is greater than the value if j.
  std::deque<size_t> best_indices;
};

template <typename INPUT, typename OUTPUT>
struct MovingMinAccumulator : MovingExtremumAccumulator<INPUT, OUTPUT> {
  MovingMinAccumulator(const ArrayRef<INPUT> &values)
      : MovingExtremumAccumulator<INPUT, OUTPUT>(values) {}

  bool Compare(INPUT a, INPUT b) { return a < b; }
};

template <typename INPUT, typename OUTPUT>
struct MovingMaxAccumulator : MovingExtremumAccumulator<INPUT, OUTPUT> {
  MovingMaxAccumulator(const ArrayRef<INPUT> &values)
      : MovingExtremumAccumulator<INPUT, OUTPUT>(values) {}

  bool Compare(INPUT a, INPUT b) { return a > b; }
};

// TODO: Revisit the MovingProductAccumulator for potential optimization to improve calculation efficiency while maintaining accuracy.
// Especially consider optimizing the Result method which recalculates the product on-demand.
template <typename INPUT, typename OUTPUT>
struct MovingProductAccumulator : public Accumulator<INPUT, OUTPUT> {
    int start_idx = 0;
    int end_idx = -1; // Initialize to -1 to indicate an empty window initially

    MovingProductAccumulator(const ArrayRef<INPUT>& values)
        : Accumulator<INPUT, OUTPUT>(values) {}

    void Add(Idx idx) override {
        // Simply move the end to the given index
        end_idx = idx;
    }

    void Remove(Idx idx) override {
        // Adjust the start index to exclude the removed value, signaling a window shift.
        start_idx = idx + 1;
    }

    OUTPUT Result() override {
        double product = 1.0;
        bool has_zero = false;

        // Calculate the product of all values inside the window
        for (int idx = start_idx; idx <= end_idx; ++idx) {
            const INPUT value = Accumulator<INPUT, OUTPUT>::values[idx];
            if (value == 0) {
                has_zero = true;
                break; // Exit early if a zero is found
            } else if (!std::isnan(value)) {
                product *= value;
            }
            // NaN values are skipped
        }

        return has_zero ? 0 : product;
    }

    void Reset()  {
        start_idx = 0;
        end_idx = -1;
    }
};



// Instantiate the "accumulate" function with and without sampling,
// and with and without variable window length.
//
// Args:
//   NAME: Name of the python and c++ function.
//   INPUT: Input value type.
//   OUTPUT: Output value type.
//   ACCUMULATOR: Accumulator class.
#define REGISTER_CC_FUNC(NAME, INPUT, OUTPUT, ACCUMULATOR)                    \
                                                                              \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,                    \
                           const py::array_t<INPUT> &evset_values,            \
                           const double window_length) {                      \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(             \
        evset_timestamps, evset_values, window_length);                       \
  }                                                                           \
                                                                              \
  py::array_t<OUTPUT> NAME(                                                   \
      const ArrayD &evset_timestamps, const py::array_t<INPUT> &evset_values, \
      const ArrayD &sampling_timestamps, const double window_length) {        \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(             \
        evset_timestamps, evset_values, sampling_timestamps, window_length);  \
  }                                                                           \
                                                                              \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,                    \
                           const py::array_t<INPUT> &evset_values,            \
                           const ArrayD &window_length) {                     \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(             \
        evset_timestamps, evset_values, window_length);                       \
  }                                                                           \
                                                                              \
  py::array_t<OUTPUT> NAME(                                                   \
      const ArrayD &evset_timestamps, const py::array_t<INPUT> &evset_values, \
      const ArrayD &sampling_timestamps, const ArrayD &window_length) {       \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(             \
        evset_timestamps, evset_values, sampling_timestamps, window_length);  \
  }

// Similar to REGISTER_CC_FUNC, but without inputs
#define REGISTER_CC_FUNC_NO_INPUT(NAME, OUTPUT, ACCUMULATOR)     \
                                                                 \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,       \
                           const double window_length) {         \
    return accumulate<double, OUTPUT, ACCUMULATOR<OUTPUT>>(      \
        evset_timestamps, evset_timestamps, window_length);      \
  }                                                              \
                                                                 \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,       \
                           const ArrayD &sampling_timestamps,    \
                           const double window_length) {         \
    return accumulate<double, OUTPUT, ACCUMULATOR<OUTPUT>>(      \
        evset_timestamps, evset_timestamps, sampling_timestamps, \
        window_length);                                          \
  }                                                              \
                                                                 \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,       \
                           const ArrayD &window_length) {        \
    return accumulate<double, OUTPUT, ACCUMULATOR<OUTPUT>>(      \
        evset_timestamps, evset_timestamps, window_length);      \
  }                                                              \
                                                                 \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,       \
                           const ArrayD &sampling_timestamps,    \
                           const ArrayD &window_length) {        \
    return accumulate<double, OUTPUT, ACCUMULATOR<OUTPUT>>(      \
        evset_timestamps, evset_timestamps, sampling_timestamps, \
        window_length);                                          \
  }

// Note: ";" are not needed for the code, but are required for our code
// formatter.

REGISTER_CC_FUNC(simple_moving_average, float, float,
                 SimpleMovingAverageAccumulator);
REGISTER_CC_FUNC(simple_moving_average, double, double,
                 SimpleMovingAverageAccumulator);

REGISTER_CC_FUNC(moving_standard_deviation, float, float,
                 MovingStandardDeviationAccumulator);
REGISTER_CC_FUNC(moving_standard_deviation, double, double,
                 MovingStandardDeviationAccumulator);

REGISTER_CC_FUNC(moving_sum, float, float, MovingSumAccumulator);
REGISTER_CC_FUNC(moving_sum, double, double, MovingSumAccumulator);
REGISTER_CC_FUNC(moving_sum, int32_t, int32_t, MovingSumAccumulator);
REGISTER_CC_FUNC(moving_sum, int64_t, int64_t, MovingSumAccumulator);

REGISTER_CC_FUNC(moving_min, float, float, MovingMinAccumulator);
REGISTER_CC_FUNC(moving_min, double, double, MovingMinAccumulator);
REGISTER_CC_FUNC(moving_min, int32_t, int32_t, MovingMinAccumulator);
REGISTER_CC_FUNC(moving_min, int64_t, int64_t, MovingMinAccumulator);

REGISTER_CC_FUNC(moving_max, float, float, MovingMaxAccumulator);
REGISTER_CC_FUNC(moving_max, double, double, MovingMaxAccumulator);
REGISTER_CC_FUNC(moving_max, int32_t, int32_t, MovingMaxAccumulator);
REGISTER_CC_FUNC(moving_max, int64_t, int64_t, MovingMaxAccumulator);

REGISTER_CC_FUNC_NO_INPUT(moving_count, int32_t, MovingCountAccumulator);

REGISTER_CC_FUNC(moving_product, float, float, MovingProductAccumulator);
REGISTER_CC_FUNC(moving_product, double, double, MovingProductAccumulator);
}  // namespace

// Register c++ functions to pybind with and without sampling,
// and with and without variable window length.
//
// Args:
//   NAME: Name of the python and c++ function.
//   INPUT: Input value type.
//   OUTPUT: Output value type.
//
#define ADD_PY_DEF(NAME, INPUT, OUTPUT)                                        \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const py::array_t<INPUT> &,          \
                          const ArrayD &, double>(&NAME),                      \
        "", py::arg("evset_timestamps").noconvert(),                           \
        py::arg("evset_values").noconvert(),                                   \
        py::arg("sampling_timestamps").noconvert(), py::arg("window_length")); \
                                                                               \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const py::array_t<INPUT> &, double>( \
            &NAME),                                                            \
        "", py::arg("evset_timestamps").noconvert(),                           \
        py::arg("evset_values").noconvert(), py::arg("window_length"));        \
                                                                               \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const py::array_t<INPUT> &,          \
                          const ArrayD &, const ArrayD &>(&NAME),              \
        "", py::arg("evset_timestamps").noconvert(),                           \
        py::arg("evset_values").noconvert(),                                   \
        py::arg("sampling_timestamps").noconvert(), py::arg("window_length")); \
                                                                               \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const py::array_t<INPUT> &,          \
                          const ArrayD &>(&NAME),                              \
        "", py::arg("evset_timestamps").noconvert(),                           \
        py::arg("evset_values").noconvert(), py::arg("window_length"));

// Similar to ADD_PY_DEF, but without inputs.
#define ADD_PY_DEF_NO_INPUT(NAME, OUTPUT)                                      \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const ArrayD &, double>(&NAME), "",  \
        py::arg("evset_timestamps").noconvert(),                               \
        py::arg("sampling_timestamps").noconvert(), py::arg("window_length")); \
                                                                               \
  m.def(#NAME, py::overload_cast<const ArrayD &, double>(&NAME), "",           \
        py::arg("evset_timestamps").noconvert(), py::arg("window_length"));    \
                                                                               \
  m.def(#NAME,                                                                 \
        py::overload_cast<const ArrayD &, const ArrayD &, const ArrayD &>(     \
            &NAME),                                                            \
        "", py::arg("evset_timestamps").noconvert(),                           \
        py::arg("sampling_timestamps").noconvert(), py::arg("window_length")); \
                                                                               \
  m.def(#NAME, py::overload_cast<const ArrayD &, const ArrayD &>(&NAME), "",   \
        py::arg("evset_timestamps").noconvert(), py::arg("window_length"));

void init_window(py::module &m) {
  ADD_PY_DEF(simple_moving_average, float, float)
  ADD_PY_DEF(simple_moving_average, double, double)

  ADD_PY_DEF(moving_standard_deviation, float, float)
  ADD_PY_DEF(moving_standard_deviation, double, double)

  ADD_PY_DEF(moving_sum, float, float)
  ADD_PY_DEF(moving_sum, double, double)
  ADD_PY_DEF(moving_sum, int32_t, int32_t)
  ADD_PY_DEF(moving_sum, int64_t, int64_t)

  ADD_PY_DEF(moving_min, float, float)
  ADD_PY_DEF(moving_min, double, double)
  ADD_PY_DEF(moving_min, int32_t, int32_t)
  ADD_PY_DEF(moving_min, int64_t, int64_t)

  ADD_PY_DEF(moving_max, float, float)
  ADD_PY_DEF(moving_max, double, double)
  ADD_PY_DEF(moving_max, int32_t, int32_t)
  ADD_PY_DEF(moving_max, int64_t, int64_t)

  ADD_PY_DEF_NO_INPUT(moving_count, int32_t)

  ADD_PY_DEF(moving_product, float, float)
  ADD_PY_DEF(moving_product, double, double)

}

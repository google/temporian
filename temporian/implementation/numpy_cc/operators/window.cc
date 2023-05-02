#include <cstdint>
#include <iostream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace py = pybind11;

typedef py::array_t<double> ArrayD;
typedef py::array_t<float> ArrayF;

// Apply TAccumulator over the data sequentially and return the aggregated
// results.
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

  TAccumulator accumulator;

  // Index of the first value in the RW.
  size_t begin_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_event; sampling_idx++) {

    accumulator.Add(v_values[sampling_idx]);

    // Note: We accumulate values in (t-window_length, t] with t=
    // v_timestamps[sampling_idx].
    const auto left_limit = v_timestamps[sampling_idx] - window_length;
    while (begin_idx < n_event && v_timestamps[begin_idx] <= left_limit) {
      accumulator.Remove(v_values[begin_idx]);
      begin_idx++;
    }

    v_output[sampling_idx] = accumulator.Result();
  }

  return output;
}

// Apply TAccumulator over the data sequentially and return the aggregated
// results.
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

  TAccumulator accumulator;

  size_t begin_idx = 0;
  size_t end_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto right_limit = v_sampling[sampling_idx];
    const auto left_limit = v_sampling[sampling_idx] - window_length;

    while (end_idx < n_event && v_timestamps[end_idx] <= right_limit) {
      accumulator.Add(v_values[end_idx]);
      end_idx++;
    }

    while (begin_idx < n_event && v_timestamps[begin_idx] <= left_limit) {
      accumulator.Remove(v_values[begin_idx]);
      begin_idx++;
    }

    v_output[sampling_idx] = accumulator.Result();
  }

  return output;
}

// Note: We only use inheritence to compile check the code.
template <typename INPUT, typename OUTPUT> struct Accumulator {
  virtual void Add(INPUT value) = 0;
  virtual void Remove(INPUT value) = 0;
  virtual OUTPUT Result() = 0;
};

template <typename INPUT, typename OUTPUT>
struct SimpleMovingAverageAccumulator : Accumulator<INPUT, OUTPUT> {

  void Add(INPUT value) override {
    if (std::isnan(value)) {
      return;
    }
    sum_values += value;
    num_values++;
  }

  void Remove(INPUT value) override {
    if (std::isnan(value)) {
      return;
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

  void Add(INPUT value) override {
    if (std::isnan(value)) {
      return;
    }
    sum_values += value;
    sum_square_values += value * value;
    num_values++;
  }

  void Remove(INPUT value) override {
    if (std::isnan(value)) {
      return;
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

template <typename INPUT, typename OUTPUT>
struct MovingCountAccumulator : Accumulator<INPUT, OUTPUT> {

  void Add(INPUT value) override {

    static_assert(std::is_same<OUTPUT, int32_t>::value,
                  "OUTPUT must be int32_t");

    // TODO: Skip for non floating point types.
    if (std::isnan(value)) {
      return;
    }
    num_values++;
  }

  void Remove(INPUT value) override {
    // TODO: Skip for non floating point types.
    if (std::isnan(value)) {
      return;
    }
    num_values--;
  }

  OUTPUT Result() override { return num_values; }

  // Number of values in the RW.
  int32_t num_values = 0;
};

template <typename INPUT, typename OUTPUT>
struct MovingSumAccumulator : Accumulator<INPUT, OUTPUT> {

  void Add(INPUT value) override {
    // TODO: Skip for non floating point types.
    if (std::isnan(value)) {
      return;
    }
    sum_values += value;
  }

  void Remove(INPUT value) override {
    // TODO: Skip for non floating point types.
    if (std::isnan(value)) {
      return;
    }
    sum_values -= value;
  }

  OUTPUT Result() override { return sum_values; }

  // Sum of the values in the rolling window (RW).
  double sum_values = 0;
};

// Instantiate the "accumulate" function with an accumulator for both float32
// and float64 precision.
//
// Args:
//   NAME: Name of the python and c++ function.
//   INPUT: Input value type.
//   OUTPUT: Output value type.
//   ACCUMULATOR: Accumulator class.
#define REGISTER_CC_FUNC(NAME, INPUT, OUTPUT, ACCUMULATOR)                     \
                                                                               \
  py::array_t<OUTPUT> NAME(const ArrayD &evset_timestamps,                     \
                           const py::array_t<INPUT> &evset_values,             \
                           const double window_length) {                       \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(              \
        evset_timestamps, evset_values, window_length);                        \
  }                                                                            \
                                                                               \
  py::array_t<OUTPUT> NAME(                                                    \
      const ArrayD &evset_timestamps, const py::array_t<INPUT> &evset_values,  \
      const ArrayD &sampling_timestamps, const double window_length) {         \
    return accumulate<INPUT, OUTPUT, ACCUMULATOR<INPUT, OUTPUT>>(              \
        evset_timestamps, evset_values, sampling_timestamps, window_length);   \
  }

REGISTER_CC_FUNC(simple_moving_average, float, float,
                 SimpleMovingAverageAccumulator)
REGISTER_CC_FUNC(simple_moving_average, double, double,
                 SimpleMovingAverageAccumulator)

REGISTER_CC_FUNC(moving_standard_deviation, float, float,
                 MovingStandardDeviationAccumulator)
REGISTER_CC_FUNC(moving_standard_deviation, double, double,
                 MovingStandardDeviationAccumulator)

REGISTER_CC_FUNC(moving_sum, float, float, MovingSumAccumulator)
REGISTER_CC_FUNC(moving_sum, double, double, MovingSumAccumulator)

REGISTER_CC_FUNC(moving_count, float, int32_t, MovingCountAccumulator)
REGISTER_CC_FUNC(moving_count, double, int32_t, MovingCountAccumulator)
REGISTER_CC_FUNC(moving_count, int32_t, int32_t, MovingCountAccumulator)
REGISTER_CC_FUNC(moving_count, int64_t, int32_t, MovingCountAccumulator)

} // namespace

// Register c++ functions to pybind with and without sampling.
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
        py::arg("evset_values").noconvert(), py::arg("window_length"));

PYBIND11_MODULE(window, m) {
  ADD_PY_DEF(simple_moving_average, float, float)
  ADD_PY_DEF(simple_moving_average, double, double)

  ADD_PY_DEF(moving_standard_deviation, float, float)
  ADD_PY_DEF(moving_standard_deviation, double, double)

  ADD_PY_DEF(moving_sum, float, float)
  ADD_PY_DEF(moving_sum, double, double)

  ADD_PY_DEF(moving_count, float, int32_t)
  ADD_PY_DEF(moving_count, double, int32_t)
  ADD_PY_DEF(moving_count, int32_t, int32_t)
  ADD_PY_DEF(moving_count, int64_t, int32_t)
}

#include <cstdint>
#include <iostream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace {
namespace py = pybind11;

typedef py::array_t<double> ArrayD;
typedef py::array_t<float> ArrayF;

template <typename T>
py::array_t<T> simple_moving_average(const ArrayD &event_timestamps,
                                     const py::array_t<T> &event_values,
                                     const double window_length) {

  // Input size
  const size_t n_event = event_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<T>(n_event);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = event_timestamps.unchecked<1>();
  auto v_values = event_values.template unchecked<1>();

  double sum_values = 0;
  int num_values = 0;
  size_t next_begin_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_event; sampling_idx++) {

    // Add to accumulator
    {
      const auto value = v_values[sampling_idx];
      if (!std::isnan(value)) {
        num_values++;
        sum_values += value;
      }
    }

    const auto left_limit = v_timestamps[sampling_idx] - window_length;
    while (next_begin_idx < n_event &&
           v_timestamps[next_begin_idx] <= left_limit) {
      if (next_begin_idx >= 0) {
        // Remove from accumulator.
        const auto value = v_values[next_begin_idx];
        if (!std::isnan(value)) {
          num_values--;
          sum_values -= value;
        }
      }
      next_begin_idx++;
    }

    const T output = (num_values > 0) ? (sum_values / num_values)
                                      : std::numeric_limits<T>::quiet_NaN();
    v_output[sampling_idx] = output;
  }

  return output;
}

// DO NOT SUBMIT: Can we merge the two?
template <typename T>
py::array_t<T> simple_moving_average(const ArrayD &event_timestamps,
                                     const py::array_t<T> &event_values,
                                     const ArrayD &sampling_timestamps,
                                     const double window_length) {
  // Input size
  const size_t n_event = event_timestamps.shape(0);
  const size_t n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto output = py::array_t<T>(n_sampling);

  auto v_output = output.template mutable_unchecked<1>();
  auto v_timestamps = event_timestamps.unchecked<1>();
  auto v_values = event_values.template unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();

  double sum_values = 0;
  int num_values = 0;

  size_t next_begin_idx = 0;
  size_t end_idx = 0;

  for (size_t sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto right_limit = v_sampling[sampling_idx];
    const auto left_limit = v_sampling[sampling_idx] - window_length;

    while (end_idx < n_event && v_timestamps[end_idx] <= right_limit) {
      // Add from accumulator.
      const auto value = v_values[end_idx];
      if (!std::isnan(value)) {
        num_values++;
        sum_values += value;
      }
      end_idx++;
    }

    while (next_begin_idx < n_event &&
           v_timestamps[next_begin_idx] <= left_limit) {
      if (next_begin_idx >= 0) {
        // Remove from accumulator.
        const auto value = v_values[next_begin_idx];
        if (!std::isnan(value)) {
          num_values--;
          sum_values -= value;
        }
      }
      next_begin_idx++;
    }

    v_output[sampling_idx] = 0;
  }

  return output;
}

} // namespace

PYBIND11_MODULE(window, m) {
  m.def(
      "simple_moving_average_float32",
      py::overload_cast<const ArrayD &, const ArrayF &, const ArrayD &, double>(
          &simple_moving_average<float>),
      "", py::arg("event_timestamps").noconvert(),
      py::arg("event_values").noconvert(),
      py::arg("sampling_timestamps").noconvert(), py::arg("window_length"));

  m.def("simple_moving_average_float32",
        py::overload_cast<const ArrayD &, const ArrayF &, double>(
            &simple_moving_average<float>),
        "", py::arg("event_timestamps").noconvert(),
        py::arg("event_values").noconvert(), py::arg("window_length"));

  m.def(
      "simple_moving_average_float64",
      py::overload_cast<const ArrayD &, const ArrayD &, const ArrayD &, double>(
          &simple_moving_average<double>),
      "", py::arg("event_timestamps").noconvert(),
      py::arg("event_values").noconvert(),
      py::arg("sampling_timestamps").noconvert(), py::arg("window_length"));

  m.def("simple_moving_average_float64",
        py::overload_cast<const ArrayD &, const ArrayD &, double>(
            &simple_moving_average<double>),
        "", py::arg("event_timestamps").noconvert(),
        py::arg("event_values").noconvert(), py::arg("window_length"));
}

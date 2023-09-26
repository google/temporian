#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "temporian/implementation/numpy_cc/operators/common.h"

namespace {
namespace py = pybind11;

std::pair<py::array_t<double>, py::array_t<double>> until_next(
    const py::array_t<double> &event_timestamps,
    const py::array_t<double> &sampling_timestamps, const double timeout) {
  // Input size
  const Idx n_event = event_timestamps.shape(0);
  const Idx n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto out_timestamps = py::array_t<double>(n_event);
  auto out_values = py::array_t<double>(n_event);

  // Access raw input / output data
  auto v_out_timestamps = out_timestamps.mutable_unchecked<1>();
  auto v_out_values = out_values.mutable_unchecked<1>();
  auto v_event = event_timestamps.unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();

  Idx next_sampling_idx = 0;
  for (Idx event_idx = 0; event_idx < n_event; event_idx++) {
    const auto t = v_event[event_idx];

    while (next_sampling_idx < n_sampling &&
           v_sampling[next_sampling_idx] < t) {
      next_sampling_idx++;
    }

    double value, timestamp;
    if (next_sampling_idx == n_sampling ||
        v_sampling[next_sampling_idx] - t > timeout) {
      timestamp = t + timeout;
      value = std::numeric_limits<double>::quiet_NaN();
    } else {
      timestamp = v_sampling[next_sampling_idx];
      value = timestamp - t;
    }

    v_out_timestamps[event_idx] = timestamp;
    v_out_values[event_idx] = value;
  }

  return std::make_pair(out_timestamps, out_values);
}

}  // namespace

void init_until_next(py::module &m) {
  m.def("until_next", &until_next, "", py::arg("event_timestamps").noconvert(),
        py::arg("sampling_timestamps").noconvert(), py::arg("timeout"));
}

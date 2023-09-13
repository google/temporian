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

py::array_t<double> filter_moving_count(
    const py::array_t<double> &event_timestamps, const double window_length) {
  // Input size
  const Idx n_event = event_timestamps.shape(0);

  // Access raw input / output data
  auto v_event = event_timestamps.unchecked<1>();

  std::vector<double> output;

  // Index of the last emitted event. If -1, no event was emitted so far.
  Idx last_emited_idx = -1;

  for (Idx event_idx = 0; event_idx < n_event; event_idx++) {
    const auto t = v_event[event_idx];
    if (last_emited_idx == -1 ||
        (t - v_event[last_emited_idx]) >= window_length) {
      // Emitting event.
      last_emited_idx = event_idx;
      output.push_back(t);
    }
  }

  return vector_to_np_array(output);
}

}  // namespace

void init_filter_moving_count(py::module &m) {
  m.def("filter_moving_count", &filter_moving_count, "",
        py::arg("event_timestamps").noconvert(), py::arg("window_length"));
}

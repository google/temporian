#include <cstdint>
#include <iostream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace {
namespace py = pybind11;

typedef size_t Idx;

std::tuple<py::array_t<Idx>, Idx>
build_sampling_idxs(const py::array_t<double> &evset_timestamps,
                    const py::array_t<double> &sampling_timestamps) {

  // Input size
  const Idx n_event = evset_timestamps.shape(0);
  const Idx n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto indices = py::array_t<Idx>(n_sampling);

  // Access raw input / output data
  auto v_idxs = indices.mutable_unchecked<1>();
  auto v_event = evset_timestamps.unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();

  // The index of the first value in "indices" that correspond to a valid
  // indice.
  Idx first_valid_idx = 0;

  Idx next_event_idx = 0;
  for (Idx sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto t = v_sampling[sampling_idx];
    while (next_event_idx < n_event && v_event[next_event_idx] <= t) {
      next_event_idx++;
    }
    v_idxs[sampling_idx] = next_event_idx - 1;
    if (next_event_idx == 0) {
      first_valid_idx = sampling_idx + 1;
    }
  }

  return std::make_tuple(indices, first_valid_idx);
}

} // namespace

PYBIND11_MODULE(sample, m) {
  m.def("build_sampling_idxs", &build_sampling_idxs, "",
        py::arg("evset_timestamps").noconvert(),
        py::arg("sampling_timestamps").noconvert());
}

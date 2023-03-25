#include <cstdint>
#include <iostream>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace {
namespace py = pybind11;

typedef size_t Indice;

std::tuple<py::array_t<Indice>, Indice>
build_sampling_indices(const py::array_t<double> &event_timestamps,
                       const py::array_t<double> &sampling_timestamps) {

  // Input size
  const Indice n_event = event_timestamps.shape(0);
  const Indice n_sampling = sampling_timestamps.shape(0);

  // Allocate output array
  auto indices = py::array_t<Indice>(n_sampling);

  // Access raw input / output data
  auto v_indice = indices.mutable_unchecked<1>();
  auto v_event = event_timestamps.unchecked<1>();
  auto v_sampling = sampling_timestamps.unchecked<1>();

  // The index of the first value in "indices" that correspond to a valid
  // indice.
  Indice first_valid_indice = 0;

  Indice next_event_idx = 0;
  for (Indice sampling_idx = 0; sampling_idx < n_sampling; sampling_idx++) {
    const auto t = v_sampling[sampling_idx];
    while (next_event_idx < n_event && v_event[next_event_idx] <= t) {
      next_event_idx++;
    }
    v_indice[sampling_idx] = next_event_idx - 1;
    if (next_event_idx == 0) {
      first_valid_indice = sampling_idx + 1;
    }
  }

  return std::make_tuple(indices, first_valid_indice);
}

} // namespace

PYBIND11_MODULE(sample, m) {
  m.def("build_sampling_indices", &build_sampling_indices, "",
        py::arg("event_timestamps").noconvert(),
        py::arg("sampling_timestamps").noconvert());
}
